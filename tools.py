# =============================================================================
# tools.py – Định nghĩa các "công cụ" (tools) mà AI Agent có thể gọi để
# truy vấn dữ liệu bán hàng từ SQLite. Mỗi hàm là một tool độc lập.
# =============================================================================

import os
import sqlite3
import pandas as pd


# -----------------------------------------------------------------------------
# Hằng số đường dẫn
# DB_PATH   : file SQLite sẽ được tạo tự động khi lần đầu chạy
# DATA_PATH : file dữ liệu nguồn (Excel hoặc CSV)
# -----------------------------------------------------------------------------
DB_PATH = "sales.db"
DATA_PATH = "data.xlsx"


# =============================================================================
# HÀM KHỞI TẠO CƠ SỞ DỮ LIỆU
# Nhiệm vụ: Đọc file Excel/CSV rồi nạp vào SQLite (chỉ chạy 1 lần duy nhất).
# Nếu file sales.db đã tồn tại thì bỏ qua, không đọc lại từ đầu.
# =============================================================================
def init_database():
    """Nạp dữ liệu từ Excel hoặc CSV vào SQLite và chuẩn hóa cột thời gian."""

    # Nếu DB đã tồn tại thì không cần khởi tạo lại → tiết kiệm thời gian
    if os.path.exists(DB_PATH):
        return

    # Tự động chọn hàm đọc file dựa vào đuôi mở rộng
    if DATA_PATH.endswith((".xlsx", ".xls")):
        df = pd.read_excel(DATA_PATH)
    else:
        # Thử UTF-8 trước, nếu lỗi encoding thì dùng ISO-8859-1 (Latin-1)
        try:
            df = pd.read_csv(DATA_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

    # Chuẩn hóa cột ngày: tạo thêm cột "InvoiceMonth" dạng "YYYY-MM"
    # để sau này có thể GROUP BY theo tháng dễ dàng
    if "InvoiceDate" in df.columns:
        parsed_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["InvoiceMonth"] = parsed_dates.dt.strftime("%Y-%m")

    # Lưu toàn bộ DataFrame vào bảng "sales" trong SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sales", conn, if_exists="replace", index=False)
    conn.close()


# =============================================================================
# TOOL 1 – inspect_dataset
# Nhiệm vụ: Cho AI biết cấu trúc bảng dữ liệu (tên cột, số dòng, dữ liệu mẫu).
# AI Agent luôn gọi tool này đầu tiên để "khám phá" dữ liệu trước khi truy vấn.
# =============================================================================
def inspect_dataset() -> dict:
    """Kiểm tra cấu trúc bảng dữ liệu: tên cột, số bản ghi, và vài dòng mẫu."""

    # Đảm bảo DB đã sẵn sàng trước khi truy vấn
    init_database()
    conn = sqlite3.connect(DB_PATH)

    # Lấy 5 dòng đầu để AI thấy dạng dữ liệu thực tế
    df_sample = pd.read_sql_query("SELECT * FROM sales LIMIT 5", conn)

    # Đếm tổng số dòng trong bảng
    total_rows = pd.read_sql_query("SELECT COUNT(*) as cnt FROM sales", conn)["cnt"][0]

    # Đếm số dòng có dữ liệu bất thường (đơn hủy, trả hàng, giá âm/0)
    invalid_rows = pd.read_sql_query(
        "SELECT COUNT(*) as cnt FROM sales WHERE Quantity <= 0 OR UnitPrice <= 0", conn
    )["cnt"][0]
    conn.close()

    # Trả về dict tóm tắt để AI đọc hiểu cấu trúc bảng
    return {
        "columns": list(df_sample.columns),                       # Danh sách tên cột
        "total_records": int(total_rows),                         # Tổng số bản ghi
        "non_positive_qty_or_price_records": int(invalid_rows),   # Số dòng bất thường
        "sample_records": df_sample.to_dict(orient="records"),    # 5 dòng mẫu
        "note": "Dataset chỉ chứa giá bán (UnitPrice), KHÔNG có cột giá vốn (Cost/COGS)."
    }


# =============================================================================
# TOOL 2 – get_total_sales
# Nhiệm vụ: Tính tổng doanh thu = SUM(Quantity * UnitPrice).
# Hỗ trợ lọc theo quốc gia và tùy chọn bao gồm/loại trừ đơn hủy.
# =============================================================================
def get_total_sales(country: str = None, include_cancellations: bool = False) -> dict:
    """Tính tổng doanh thu (Quantity * UnitPrice) và tổng số lượng bán ra."""

    init_database()
    conn = sqlite3.connect(DB_PATH)

    # Xây dựng điều kiện WHERE linh hoạt dựa vào tham số đầu vào
    conditions = []
    params = []

    # Mặc định loại bỏ đơn hủy / trả hàng (Quantity hoặc UnitPrice âm/bằng 0)
    if not include_cancellations:
        conditions.append("Quantity > 0 AND UnitPrice > 0")

    # Nếu người dùng lọc theo quốc gia cụ thể, thêm điều kiện so sánh
    # dùng LOWER() để không phân biệt hoa thường
    if country:
        conditions.append("LOWER(Country) = LOWER(?)")
        params.append(country)

    # Ghép các điều kiện thành mệnh đề WHERE hoàn chỉnh
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Truy vấn tổng hợp: số giao dịch, tổng sản phẩm bán, tổng doanh thu
    query = f"""
        SELECT 
            COUNT(*) as valid_transactions,
            SUM(Quantity) as total_units_sold,
            ROUND(SUM(Quantity * UnitPrice), 2) as total_sales_value
        FROM sales
        {where_clause}
    """
    res = pd.read_sql_query(query, conn, params=params).to_dict(orient="records")[0]
    conn.close()

    # Nếu không có dữ liệu khớp bộ lọc, trả về thông báo rỗng
    if not res["valid_transactions"]:
        return {"status": "empty", "message": f"Không tìm thấy dữ liệu giao dịch cho bộ lọc country={country}."}

    # Ghi chú giả định đã áp dụng vào kết quả để AI thông báo cho người dùng
    res["assumption_applied"] = (
        "Đã loại bỏ các giao dịch hủy/trả hàng (Quantity <= 0) và đơn giá <= 0."
        if not include_cancellations else "Đã bao gồm cả các giao dịch âm/đơn hủy."
    )
    return res


# =============================================================================
# TOOL 3 – get_top_rankings
# Nhiệm vụ: Xếp hạng Top N theo quốc gia hoặc sản phẩm,
# đo theo doanh thu (sales) hoặc số lượng bán (quantity).
# Kết quả kèm theo "chart_type: bar" để giao diện tự động vẽ biểu đồ cột.
# =============================================================================
def get_top_rankings(group_by: str = "Country", metric: str = "sales", top_n: int = 5) -> dict:
    """Xếp hạng Top N theo Quốc gia (Country) hoặc Sản phẩm (Description/StockCode)."""

    init_database()

    # Chỉ cho phép nhóm theo các cột hợp lệ để tránh SQL injection
    allowed_groups = {"Country", "Description", "StockCode"}
    if group_by not in allowed_groups:
        return {"error": f"Chỉ hỗ trợ nhóm theo các cột: {allowed_groups}"}

    # Chọn công thức SQL và nhãn kết quả tùy theo metric
    metric_sql = "ROUND(SUM(Quantity * UnitPrice), 2)" if metric == "sales" else "SUM(Quantity)"
    metric_label = "total_sales_value" if metric == "sales" else "total_quantity_sold"

    conn = sqlite3.connect(DB_PATH)

    # Truy vấn xếp hạng: nhóm theo cột được chỉ định, lấy Top N cao nhất
    query = f"""
        SELECT 
            {group_by} as label,
            {metric_sql} as value
        FROM sales
        WHERE Quantity > 0 AND UnitPrice > 0 AND {group_by} IS NOT NULL
        GROUP BY {group_by}
        ORDER BY value DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(top_n,))
    conn.close()

    # Trả về kết quả kèm metadata để giao diện biết cần vẽ biểu đồ cột (bar)
    return {
        "group_by": group_by,
        "metric": metric_label,
        "top_n": top_n,
        "results": df.to_dict(orient="records"),    # Dữ liệu cho biểu đồ
        "chart_type": "bar",                        # Loại biểu đồ: cột
        "assumption_applied": "Chỉ tính các giao dịch hợp lệ (Quantity > 0 và UnitPrice > 0)."
    }


# =============================================================================
# TOOL 4 – get_sales_by_month
# Nhiệm vụ: Thống kê doanh thu từng tháng theo chuỗi thời gian.
# Kết quả kèm theo "chart_type: line" để giao diện vẽ biểu đồ đường.
# =============================================================================
def get_sales_by_month() -> dict:
    """Thống kê tổng doanh thu theo từng tháng (YYYY-MM)."""

    init_database()
    conn = sqlite3.connect(DB_PATH)

    # Nhóm theo cột InvoiceMonth (đã được tạo khi khởi tạo DB),
    # sắp xếp tăng dần để biểu đồ hiển thị đúng thứ tự thời gian
    query = """
        SELECT 
            InvoiceMonth as label,
            ROUND(SUM(Quantity * UnitPrice), 2) as value,
            SUM(Quantity) as total_units
        FROM sales
        WHERE Quantity > 0 AND UnitPrice > 0 AND InvoiceMonth IS NOT NULL
        GROUP BY InvoiceMonth
        ORDER BY InvoiceMonth ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Trả về kết quả kèm metadata để giao diện biết cần vẽ biểu đồ đường (line)
    return {
        "results": df.to_dict(orient="records"),    # Dữ liệu cho biểu đồ
        "chart_type": "line",                       # Loại biểu đồ: đường
        "assumption_applied": "Doanh thu theo tháng chỉ tính các giao dịch có Quantity > 0 và UnitPrice > 0."
    }
