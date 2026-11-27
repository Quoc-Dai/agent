# =============================================================================
# tools.py – Định nghĩa các "công cụ" (tools) mà AI Agent có thể gọi để
# truy vấn dữ liệu bán hàng từ DuckDB. Mỗi hàm là một tool độc lập.
#
# ⚡ DuckDB thay thế SQLite vì:
#   - Kiến trúc OLAP (columnar) → GROUP BY / SUM / COUNT nhanh hơn 5-10x
#   - Đọc trực tiếp từ Parquet → không cần load toàn bộ dữ liệu vào RAM
#   - In-process, không cần server, cài đặt bằng pip install duckdb
# =============================================================================

import glob
import json
import os
import time
import duckdb
import pandas as pd


# -----------------------------------------------------------------------------
# Hằng số đường dẫn
# DB_PATH      : file DuckDB sẽ được tạo tự động khi lần đầu chạy
# PARQUET_PATH : file Parquet trung gian (nhanh hơn đọc Excel trực tiếp)
# DATA_DIR     : thư mục chứa file dữ liệu nguồn (Excel hoặc CSV)
# META_PATH    : file JSON lưu metadata (tên file nguồn + thời gian sửa đổi)
#                dùng để so sánh khi nào cần rebuild database
# -----------------------------------------------------------------------------
DB_PATH      = "sales.duckdb"
PARQUET_PATH = "sales.parquet"
DATA_DIR     = "."
META_PATH    = "sales.meta.json"


# =============================================================================
# HÀM TÌM FILE DỮ LIỆU NGUỒN
# Tự động quét thư mục hiện tại tìm file Excel/CSV mới nhất,
# hoặc dùng đường dẫn chỉ định từ tham số.
# =============================================================================
def _find_data_file(data_path: str = None) -> str:  # Tìm file dữ liệu nguồn (Excel/CSV) mới nhất hoặc theo đường dẫn chỉ định
    """Tìm file dữ liệu nguồn (Excel/CSV) để nạp vào database.

    Args:
        data_path: Đường dẫn file chỉ định (nếu có). Nếu None sẽ quét DATA_DIR lấy file mới nhất.

    Returns:
        Đường dẫn tới file dữ liệu nguồn hợp lệ.
    """
    if data_path and os.path.exists(data_path):
        return data_path

    # Quét tất cả file Excel/CSV trong thư mục, chọn file mới nhất (theo thời gian sửa đổi)
    patterns = ["*.xlsx", "*.xls", "*.csv"]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(DATA_DIR, pat)))

    # Loại bỏ file parquet và file tạm
    candidates = [f for f in candidates if not f.endswith(".parquet")]

    if not candidates:
        raise FileNotFoundError(
            f"❌ Không tìm thấy file dữ liệu (.xlsx, .xls, .csv) trong thư mục '{os.path.abspath(DATA_DIR)}'.\n"
            "Hãy đặt file dữ liệu vào thư mục dự án hoặc upload qua giao diện."
        )

    # Chọn file có thời gian sửa đổi mới nhất
    best = max(candidates, key=os.path.getmtime)
    print(f"📂 [Database] Tìm thấy {len(candidates)} file dữ liệu, chọn: `{os.path.basename(best)}` (mới nhất)")
    return best


# =============================================================================
# HÀM KIỂM TRA CẦN REBUILD DATABASE KHÔNG
# So sánh metadata (tên file + thời gian sửa) của file nguồn hiện tại
# với metadata đã lưu lần build trước. Nếu khác nhau → cần rebuild.
# =============================================================================
def _needs_rebuild(data_path: str) -> bool:  # Kiểm tra xem DB có cần build lại do thiếu file hoặc dữ liệu nguồn thay đổi không
    """Kiểm tra xem có cần xây dựng lại database DuckDB hay không.

    So sánh đường dẫn và thời gian sửa đổi (mtime) của file nguồn với metadata đã lưu.

    Args:
        data_path: Đường dẫn tới file dữ liệu nguồn hiện tại.

    Returns:
        True nếu chưa có DB/metadata hoặc file nguồn đã thay đổi; False nếu có thể dùng lại DB cũ.
    """
    # Chưa có database → cần build
    if not os.path.exists(DB_PATH):
        return True

    # Chưa có metadata → không biết build từ file nào → rebuild cho an toàn
    if not os.path.exists(META_PATH):
        return True

    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # So sánh tên file và thời gian sửa đổi
        current_mtime = os.path.getmtime(data_path)
        if meta.get("source_file") != os.path.abspath(data_path):
            print(f"🔄 [Database] File nguồn đã thay đổi: `{meta.get('source_file')}` → `{data_path}`")
            return True
        if meta.get("source_mtime") != current_mtime:
            print(f"🔄 [Database] File `{os.path.basename(data_path)}` đã được cập nhật → cần rebuild")
            return True

        return False
    except (json.JSONDecodeError, KeyError):
        return True


def _save_meta(data_path: str, row_count: int):  # Lưu metadata (tên file, thời gian sửa, số dòng) vào sales.meta.json
    """Lưu thông tin metadata (tên file, mtime, số dòng, thời gian build) sau khi khởi tạo DB.

    Args:
        data_path: Đường dẫn file dữ liệu nguồn đã nạp.
        row_count: Tổng số dòng dữ liệu đã xử lý.
    """
    meta = {
        "source_file": os.path.abspath(data_path),
        "source_mtime": os.path.getmtime(data_path),
        "row_count": row_count,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"💾 [Database] Đã lưu metadata: {os.path.basename(data_path)} ({row_count:,} dòng)")


def get_data_info() -> dict:  # Lấy thông tin file dữ liệu đang dùng từ metadata để hiển thị lên UI
    """Đọc metadata để lấy thông tin file dữ liệu đang sử dụng (hiển thị trên UI).

    Returns:
        Dict chứa `source_file`, `row_count`, `built_at` hoặc None nếu chưa có metadata.
    """
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "source_file": os.path.basename(meta.get("source_file", "N/A")),
                "row_count": meta.get("row_count", 0),
                "built_at": meta.get("built_at", "N/A"),
            }
        except Exception:
            pass
    return None


# =============================================================================
# HÀM XÓA DATABASE CŨ ĐỂ REBUILD
# Đóng connection hiện tại, xóa file DuckDB + Parquet + metadata
# =============================================================================
def reset_database():  # Đóng kết nối DB và xóa các file sales.duckdb, sales.parquet, sales.meta.json cũ
    """Đóng kết nối DuckDB hiện tại và xóa các file DB, Parquet, metadata cũ để rebuild."""
    global _SHARED_CONN
    if _SHARED_CONN is not None:
        try:
            _SHARED_CONN.close()
        except Exception:
            pass
        _SHARED_CONN = None

    for f in [DB_PATH, PARQUET_PATH, META_PATH]:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ [Database] Đã xóa `{f}`")


# =============================================================================
# HÀM KHỞI TẠO CƠ SỞ DỮ LIỆU
# Tự động phát hiện file mới → rebuild nếu cần
# Bước 1 – Đọc Excel/CSV → lưu Parquet (nén, đọc cực nhanh lần sau)
# Bước 2 – Nạp Parquet vào DuckDB, chuẩn hóa cột thời gian
# =============================================================================
def init_database(data_path: str = None):  # Khởi tạo ETL: đọc Excel/CSV -> làm sạch bằng Pandas -> xuất Parquet -> nạp DuckDB
    """Nạp dữ liệu từ Excel/CSV → Parquet → DuckDB với chuẩn hóa sâu & tối ưu hóa tối đa.
    
    Args:
        data_path: Đường dẫn file dữ liệu cụ thể. Nếu None, tự tìm file mới nhất.
    """

    # Tìm file dữ liệu nguồn
    source_file = _find_data_file(data_path)

    # Kiểm tra có cần rebuild không (file mới hoặc file đã cập nhật)
    if not _needs_rebuild(source_file):
        return

    # Nếu cần rebuild → xóa database cũ trước
    reset_database()

    print("🚀 [Database] Khởi tạo DuckDB & Parquet từ nguồn dữ liệu...")
    t_start = time.time()

    # --- Bước 1: Đọc file nguồn ---
    print(f"📂 [Database] Đang đọc file gốc `{os.path.basename(source_file)}`...")
    if source_file.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source_file)
    else:
        try:
            df = pd.read_csv(source_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(source_file, encoding="ISO-8859-1")
    print(f"   Đã đọc xong {len(df):,} dòng.")

    # --- Bước 2: Chuẩn hóa cơ bản các chuỗi văn bản ---
    print("🧹 [Database] Đang chuẩn hóa văn bản & xử lý khoảng trắng...")
    if "Country" in df.columns:
        df["Country"] = df["Country"].astype(str).str.strip().replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    if "StockCode" in df.columns:
        df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    if "InvoiceNo" in df.columns:
        df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

    # 2.1 Chuẩn hóa cột ngày & tháng
    if "InvoiceDate" in df.columns:
        parsed_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["InvoiceMonth"] = parsed_dates.dt.strftime("%Y-%m")

    # 2.2 Xử lý Missing & chuẩn hóa cột CustomerID
    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].apply(
            lambda x: "GUEST" if pd.isna(x) or str(x).strip().lower() in ["nan", "none", ""] 
            else str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x).strip()
        )

    # 2.3 Phục hồi Description bị thiếu thông minh (Smart Imputation bằng StockCode)
    print("🧠 [Database] Đang phục hồi Description thiếu từ bản đồ StockCode...")
    if "Description" in df.columns and "StockCode" in df.columns:
        # Bước a: Lấy bản đồ các StockCode có tên hợp lệ
        valid_desc_df = df[
            df["Description"].notna() & 
            (~df["Description"].astype(str).str.strip().str.upper().isin(["NAN", "NONE", "", "?"])) &
            (~df["Description"].astype(str).str.strip().str.startswith("?"))
        ][["StockCode", "Description"]].drop_duplicates(subset=["StockCode"])
        
        stock_to_desc = dict(zip(valid_desc_df["StockCode"], valid_desc_df["Description"].str.strip().str.upper()))

        def smart_clean_desc(row):  # Phục hồi tên sản phẩm (Description) bị thiếu bằng cách tra cứu theo StockCode
            """Chuẩn hóa và phục hồi tên sản phẩm (Description) bị thiếu dựa theo mã StockCode."""
            desc = row["Description"]
            stock = row["StockCode"]
            if pd.isna(desc):
                return stock_to_desc.get(stock, "UNKNOWN")
            s = str(desc).strip().upper()
            if s in ["NAN", "NONE", "", "?"] or s.startswith("?"):
                return stock_to_desc.get(stock, "UNKNOWN")
            return s

        df["Description"] = df.apply(smart_clean_desc, axis=1)

    # --- Bước 3: Ép kiểu dữ liệu tối ưu kích thước bộ nhớ (Downcasting) ---
    print("⚡ [Database] Ép kiểu dữ liệu tối ưu bộ nhớ (Downcasting)...")
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], downcast="integer")
    if "UnitPrice" in df.columns:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], downcast="float")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    # --- Bước 4: Lưu ra Parquet (nén snappy, đọc lại cực nhanh) ---
    print(f"📦 [Database] Đang xuất sang `{PARQUET_PATH}`...")
    df.to_parquet(PARQUET_PATH, index=False, compression="snappy")

    # --- Bước 5: Nạp Parquet vào DuckDB & Tối ưu hóa bảng ---
    print(f"🦆 [Database] Đang nạp vào DuckDB `{DB_PATH}` & tạo Zone Maps + Indexes...")
    conn = duckdb.connect(DB_PATH)
    conn.execute(f"""
        CREATE TABLE sales AS
        SELECT 
            InvoiceNo,
            StockCode,
            Description,
            Quantity,
            InvoiceDate,
            UnitPrice,
            CustomerID,
            Country,
            InvoiceMonth,
            ROUND(Quantity * UnitPrice, 2) AS SalesAmount,
            (Quantity > 0 AND UnitPrice > 0) AS IsValidSale,
            (Quantity < 0 OR InvoiceNo LIKE 'C%') AS IsCancelled
        FROM read_parquet('{PARQUET_PATH}')
        ORDER BY Country, InvoiceMonth, Description
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_country ON sales(Country);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_month ON sales(InvoiceMonth);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_valid ON sales(IsValidSale);")
    conn.close()

    # Lưu metadata để lần sau biết file nào đã build
    _save_meta(source_file, len(df))

    print(f"✨ [Database] Chuẩn hóa & Tối ưu hoàn tất xuất sắc trong {time.time() - t_start:.2f}s!")
    print(f"✨ [Database] Khởi tạo hoàn tất trong {time.time() - t_start:.2f}s!")


# =============================================================================
# HELPER: Giữ kết nối tĩnh (Singleton) tránh mở/đóng liên tục gây chậm
# =============================================================================
_SHARED_CONN = None

def _connect() -> duckdb.DuckDBPyConnection:  # Trả về kết nối DuckDB dùng chung ở chế độ chỉ đọc (Singleton)
    """Khởi tạo DB nếu cần và trả về kết nối DuckDB dùng chung ở chế độ chỉ đọc (Singleton)."""
    global _SHARED_CONN
    init_database()
    if _SHARED_CONN is None:
        _SHARED_CONN = duckdb.connect(DB_PATH, read_only=True)
    return _SHARED_CONN


# =============================================================================
# TOOL 1 – inspect_dataset
# Nhiệm vụ: Cho AI biết cấu trúc bảng dữ liệu (tên cột, số dòng, dữ liệu mẫu).
# AI Agent luôn gọi tool này đầu tiên để "khám phá" dữ liệu trước khi truy vấn.
# =============================================================================
def inspect_dataset() -> dict:  # Tool 1: Kiểm tra cấu trúc bảng sales (tên cột, tổng dòng, dòng lỗi, 5 dòng mẫu)
    """Kiểm tra cấu trúc bảng `sales`: danh sách cột, tổng số dòng, số dòng bất thường và 5 dòng mẫu."""

    conn = _connect()

    # Lấy 5 dòng đầu để AI thấy dạng dữ liệu thực tế
    df_sample = conn.execute("SELECT * FROM sales LIMIT 5").df()

    # Đếm tổng số dòng
    total_rows = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]

    # Đếm số dòng bất thường (đơn hủy, trả hàng, giá âm/0)
    invalid_rows = conn.execute(
        "SELECT COUNT(*) FROM sales WHERE IsValidSale = FALSE"
    ).fetchone()[0]

    return {
        "columns": list(df_sample.columns),                       # Danh sách tên cột
        "total_records": int(total_rows),                         # Tổng số bản ghi
        "non_positive_qty_or_price_records": int(invalid_rows),   # Số dòng bất thường
        "sample_records": df_sample.to_dict(orient="records"),    # 5 dòng mẫu
        "currency": "USD",
        "note": "Dataset sử dụng đơn vị tiền tệ cố định là USD ($). Dataset chỉ chứa giá bán (UnitPrice), KHÔNG có cột giá vốn (Cost/COGS)."
    }


# =============================================================================
# TOOL 2 – get_total_sales
# Nhiệm vụ: Tính tổng doanh thu = SUM(SalesAmount).
# Hỗ trợ lọc theo quốc gia và tùy chọn bao gồm/loại trừ đơn hủy.
# =============================================================================
def get_total_sales(country: str = None, include_cancellations: bool = False) -> dict:  # Tool 2: Tính tổng doanh thu (USD), số giao dịch và số lượng bán
    """Tính tổng doanh thu (USD), số giao dịch và tổng số lượng bán ra.

    Args:
        country: Tên quốc gia cần lọc (None nếu tính toàn cầu).
        include_cancellations: True để tính cả đơn hủy/âm; False để chỉ tính giao dịch hợp lệ.

    Returns:
        Dict chứa kết quả thống kê và đơn vị tiền tệ USD.
    """

    conn = _connect()

    conditions = []
    params = []

    if not include_cancellations:
        conditions.append("IsValidSale = TRUE")

    if country:
        conditions.append("LOWER(Country) = LOWER($1)")
        params.append(country)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            COUNT(*)                    AS valid_transactions,
            SUM(Quantity)               AS total_units_sold,
            ROUND(SUM(SalesAmount), 2)  AS total_sales_value
        FROM sales
        {where_clause}
    """
    res = conn.execute(query, params).df().to_dict(orient="records")[0]

    if not res["valid_transactions"]:
        return {"status": "empty", "message": f"Không tìm thấy dữ liệu giao dịch cho bộ lọc country={country}."}

    res["currency"] = "USD"
    res["assumption_applied"] = (
        "Đã loại bỏ các giao dịch hủy/trả hàng (Quantity <= 0) và đơn giá <= 0. Doanh thu tính theo USD ($)."
        if not include_cancellations else "Đã bao gồm cả các giao dịch âm/đơn hủy. Doanh thu tính theo USD ($)."
    )
    return res


# =============================================================================
# TOOL 3 – get_top_rankings
# Nhiệm vụ: Xếp hạng Top N theo quốc gia hoặc sản phẩm,
# đo theo doanh thu (sales) hoặc số lượng bán (quantity).
# Kết quả kèm theo "chart_type: bar" để giao diện tự động vẽ biểu đồ cột.
# =============================================================================
def get_top_rankings(group_by: str = "Country", metric: str = "sales", top_n: int = 5) -> dict:  # Tool 3: Xếp hạng Top N Quốc gia/Sản phẩm và trả về dữ liệu vẽ biểu đồ cột
    """Xếp hạng Top N theo Quốc gia hoặc Sản phẩm để trả về số liệu và dữ liệu vẽ biểu đồ cột.

    Args:
        group_by: Cột gom nhóm (`Country`, `Description`, hoặc `StockCode`).
        metric: Tiêu chí xếp hạng (`sales` theo doanh thu USD, hoặc `quantity` theo số lượng).
        top_n: Số lượng thứ hạng cao nhất cần lấy.

    Returns:
        Dict chứa danh sách xếp hạng và cấu hình `chart_type: bar`.
    """

    # Whitelist để tránh SQL injection
    allowed_groups = {"Country", "Description", "StockCode"}
    if group_by not in allowed_groups:
        return {"error": f"Chỉ hỗ trợ nhóm theo các cột: {allowed_groups}"}

    metric_sql   = "ROUND(SUM(SalesAmount), 2)" if metric == "sales" else "SUM(Quantity)"
    metric_label = "total_sales_value"          if metric == "sales" else "total_quantity_sold"

    conn = _connect()
    query = f"""
        SELECT
            {group_by}   AS label,
            {metric_sql} AS value
        FROM sales
        WHERE IsValidSale = TRUE AND {group_by} IS NOT NULL
        GROUP BY {group_by}
        ORDER BY value DESC
        LIMIT {int(top_n)}
    """
    df = conn.execute(query).df()

    return {
        "group_by":          group_by,
        "metric":            metric_label,
        "top_n":             top_n,
        "currency":          "USD" if metric == "sales" else None,
        "results":           df.to_dict(orient="records"),
        "chart_type":        "bar",
        "assumption_applied": "Chỉ tính các giao dịch hợp lệ (Quantity > 0 và UnitPrice > 0)."
    }


# =============================================================================
# TOOL 4 – get_sales_by_month
# Nhiệm vụ: Thống kê doanh thu từng tháng theo chuỗi thời gian.
# Kết quả kèm theo "chart_type: line" để giao diện vẽ biểu đồ đường.
# =============================================================================
def get_sales_by_month() -> dict:  # Tool 4: Thống kê doanh thu theo từng tháng (YYYY-MM) và trả về dữ liệu vẽ biểu đồ đường
    """Thống kê doanh thu và số lượng bán theo từng tháng (YYYY-MM), kèm tháng cao/thấp nhất và `chart_type: line`."""

    conn = _connect()
    query = """
        SELECT
            InvoiceMonth               AS label,
            ROUND(SUM(SalesAmount), 2) AS value,
            SUM(Quantity)              AS total_units
        FROM sales
        WHERE IsValidSale = TRUE AND InvoiceMonth IS NOT NULL
        GROUP BY InvoiceMonth
        ORDER BY InvoiceMonth ASC
    """
    df = conn.execute(query).df()

    # Tìm tháng có doanh thu cao nhất & thấp nhất
    highest_row = df.loc[df["value"].idxmax()].to_dict() if not df.empty else None
    lowest_row = df.loc[df["value"].idxmin()].to_dict() if not df.empty else None

    return {
        "results":           df.to_dict(orient="records"),
        "currency":          "USD",
        "chart_type":        "line",
        "highest_month":     highest_row,
        "lowest_month":      lowest_row,
        "assumption_applied": "Doanh thu theo tháng chỉ tính các giao dịch có Quantity > 0 và UnitPrice > 0."
    }


# =============================================================================
# TOOL 5 – get_sales_by_year
# Nhiệm vụ: Thống kê doanh thu theo từng năm (2010, 2011).
# =============================================================================
def get_sales_by_year() -> dict:  # Tool 5: Thống kê doanh thu và số lượng bán theo từng năm (YYYY)
    """Thống kê tổng doanh thu và số lượng bán theo từng năm (YYYY), kèm năm cao/thấp nhất và `chart_type: bar`."""

    conn = _connect()
    query = """
        SELECT
            SUBSTR(InvoiceMonth, 1, 4) AS label,
            ROUND(SUM(SalesAmount), 2) AS value,
            SUM(Quantity)              AS total_units
        FROM sales
        WHERE IsValidSale = TRUE AND InvoiceMonth IS NOT NULL
        GROUP BY label
        ORDER BY label ASC
    """
    df = conn.execute(query).df()

    highest_row = df.loc[df["value"].idxmax()].to_dict() if not df.empty else None
    lowest_row = df.loc[df["value"].idxmin()].to_dict() if not df.empty else None

    return {
        "results":           df.to_dict(orient="records"),
        "currency":          "USD",
        "chart_type":        "bar",
        "highest_year":      highest_row,
        "lowest_year":       lowest_row,
        "assumption_applied": "Doanh thu theo năm chỉ tính các giao dịch có Quantity > 0 và UnitPrice > 0."
    }
