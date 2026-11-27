# =============================================================================
# agent.py – "Bộ não" của AI Agent: quản lý truy vấn dữ liệu bán hàng.
# Sử dụng Automatic Function Calling chính hãng của Google GenAI SDK
# =============================================================================

import functools
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Import các tools từ tools.py
from tools import (
    inspect_dataset,
    get_total_sales,
    get_top_rankings,
    get_sales_by_month,
    get_sales_by_year,
)


# Endpoint proxy/gateway tùy chỉnh (để trống nếu dùng trực tiếp Google API)
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "http://ai.duocvinhgia.com").strip()

# Danh sách model dự phòng mặc định (ưu tiên gemini-3-flash theo gateway cấu hình)
DEFAULT_FALLBACK_MODELS = [
    "gemini-3-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.8-flash",
    "gemini-2.0-flash",
]

# Thứ tự ưu tiên model khi tự động lọc từ API key
PREFERRED_MODEL_ORDER = [
    "gemini-3-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

_MODELS_CACHE = {}

SYSTEM_PROMPT = """
Bạn là một AI Chuyên viên Phân tích Dữ liệu Bán hàng.
Dữ liệu nằm trong bảng `sales` (gồm 541,909 dòng giao dịch từ 12/2010 đến 12/2011).
Đơn vị tiền tệ cố định duy nhất của toàn bộ hệ thống là USD ($).

Các công cụ phân tích có sẵn:
1. `get_total_sales(country, include_cancellations)`: Dùng khi hỏi tổng doanh thu, số lượng bán ra (toàn cầu hoặc theo quốc gia).
2. `get_sales_by_month()`: Dùng khi hỏi doanh thu theo tháng, xu hướng bán hàng, hoặc tháng nào cao nhất / thấp nhất.
3. `get_sales_by_year()`: Dùng khi hỏi doanh số/doanh thu theo năm, năm nào bán nhiều nhất/ít nhất.
4. `get_top_rankings(group_by, metric, top_n)`: Dùng khi hỏi Top sản phẩm (Description) hoặc Top quốc gia (Country) theo doanh thu hoặc số lượng.
5. `inspect_dataset()`: Dùng khi hỏi về cấu trúc bảng dữ liệu, số lượng dòng, các cột.

Quy tắc:
- ĐƠN VỊ TIỀN TỆ CỐ ĐỊNH: Mọi số liệu liên quan đến tiền bạc (doanh thu, giá bán, tổng tiền, UnitPrice, total_sales_value) ĐỀU PHẢI dùng đơn vị tiền tệ cố định là USD ($).
- Khi trả lời về số tiền, luôn hiển thị ký hiệu `$` hoặc `USD` kèm định dạng số dễ đọc (ví dụ: `$1,234,567.89` hoặc `1,234,567.89 USD`). Tuyệt đối không dùng £ (Bảng Anh) hay các đơn vị tiền tệ khác.
- Hãy sử dụng công cụ phù hợp để lấy số liệu thực tế trước khi trả lời.
- Tuyệt đối không tự bịa đặt số liệu.
- Nếu người dùng hỏi lợi nhuận (Profit), hãy giải thích là dữ liệu không có giá vốn nên không tính được.
- Trả lời rõ ràng, trực diện vào câu hỏi và nêu ngắn gọn con số cụ thể.
"""

TOOL_MAP = {
    "inspect_dataset": inspect_dataset,
    "get_total_sales": get_total_sales,
    "get_top_rankings": get_top_rankings,
    "get_sales_by_month": get_sales_by_month,
    "get_sales_by_year": get_sales_by_year,
}

_CLIENT_CACHE = {}

def get_client(api_key: str, force_new: bool = False) -> genai.Client:  # Khởi tạo và cache client kết nối Gemini API (hỗ trợ custom base_url)
    """Khởi tạo Client kết nối với Gemini API (hỗ trợ custom base_url v1beta).
    LƯU Ý QUAN TRỌNG: HttpOptions.timeout của google-genai SDK tính bằng MILIGIÂY (ms).
    120,000 ms = 120 giây (đủ thời gian cho các chu trình gọi công cụ và phân tích).
    """
    if force_new or api_key not in _CLIENT_CACHE:
        http_opts = (
            types.HttpOptions(base_url=GEMINI_BASE_URL, api_version="v1beta", timeout=120000.0)
            if GEMINI_BASE_URL
            else types.HttpOptions(timeout=120000.0)
        )
        _CLIENT_CACHE[api_key] = genai.Client(
            api_key=api_key,
            http_options=http_opts,
        )
    return _CLIENT_CACHE[api_key]


def get_models_for_key(client: genai.Client, api_key: str) -> list[str]:  # Lấy danh sách model Gemini khả dụng theo thứ tự ưu tiên cho API key hiện tại
    """Lấy danh sách model Gemini hỗ trợ `generateContent` theo thứ tự ưu tiên.

    Nếu đang cấu hình `GEMINI_BASE_URL` (proxy/gateway), dùng thẳng `DEFAULT_FALLBACK_MODELS`.
    Ngược lại sẽ truy vấn `client.models.list()`, sắp xếp theo `PREFERRED_MODEL_ORDER` và cache theo API key.
    """
    # Nếu đang dùng custom proxy/gateway thì gọi thẳng danh sách model đã cấu hình (ưu tiên gemini-3-flash)
    if GEMINI_BASE_URL:
        return DEFAULT_FALLBACK_MODELS

    if api_key in _MODELS_CACHE:
        return _MODELS_CACHE[api_key]

    try:
        supported = []
        for m in client.models.list():
            name = (m.name or "").removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" in name and (not actions or "generateContent" in actions):
                supported.append(name)

        if supported:
            # Ưu tiên theo PREFERRED_MODEL_ORDER (bản nhẹ nhất trước), rồi mới tới các bản flash/lite khác
            ordered = [m for m in PREFERRED_MODEL_ORDER if m in supported]
            extras = [
                m for m in supported
                if m not in ordered and ("flash" in m or "lite" in m)
            ]
            models = ordered + extras
            if models:
                _MODELS_CACHE[api_key] = models
                print(f"🤖 [Models cho key hiện tại]: {models}")
                return models
    except Exception as e:
        print(f"⚠️ Không thể lấy danh sách model từ API ({e}), dùng danh sách mặc định.")

    return DEFAULT_FALLBACK_MODELS


def run_agent(user_message: str, chat_history: list, api_key: str, status_cb=None, **kwargs) -> dict:  # Chạy AI Agent: điều phối vòng lặp gọi tool thủ công (tối đa 6 bước) và fallback model
    """Điều phối AI Agent xử lý câu hỏi người dùng thông qua vòng lặp gọi tool thủ công và fallback model.

    Args:
        user_message: Câu hỏi ngôn ngữ tự nhiên từ người dùng.
        chat_history: Lịch sử hội thoại trước đó (`[{"role": ..., "content": ...}]`).
        api_key: Gemini API Key dùng để khởi tạo/truy xuất client.
        status_cb: Callback cập nhật trạng thái xử lý theo thời gian thực lên giao diện (tùy chọn).

    Returns:
        Dict gồm `answer` (câu trả lời văn bản), `tool_logs` (lịch sử gọi tool) và `chart` (dữ liệu vẽ biểu đồ nếu có).
    """

    t_start = time.time()
    print("\n" + "=" * 60)
    print(f"📩 [USER]: {user_message}")
    print(f"⏰ [Bắt đầu]: {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    if status_cb:
        status_cb("Đang phân tích câu hỏi và truy vấn dữ liệu...")

    client = get_client(api_key)

    # Biến lưu trữ logs và chart thu được trong quá trình chạy
    executed_tools = []
    chart_payload = None

    # Tạo Wrapper để ghi lại logs và chart khi SDK tự động gọi tool
    def make_logged_tool(name, func):  # Bọc tool gốc để tự động ghi log và bắt dữ liệu biểu đồ
        """Bọc hàm tool gốc để ghi log tham số/kết quả, đo thời gian chạy và trích xuất payload biểu đồ."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # Hàm thực thi thay thế: đo thời gian chạy, lưu tool_logs và trích xuất chart_payload
            """Hàm bao thực thi tool, lưu `executed_tools` và cập nhật `chart_payload` nếu kết quả có biểu đồ."""
            nonlocal chart_payload
            print(f"🛠️ [SDK Auto-Call] Tool `{name}` đang chạy với args: {kwargs}...")
            if status_cb:
                status_cb(f"Đang thực thi công cụ `{name}`...")

            t_tool = time.time()
            res = func(*args, **kwargs)
            duration = time.time() - t_tool
            print(f"   ⚡ Xong trong {duration*1000:.2f} ms")

            executed_tools.append({
                "tool": name,
                "args": kwargs,
                "result": res
            })

            if isinstance(res, dict) and "chart_type" in res and "results" in res:
                chart_payload = res

            if status_cb:
                status_cb("Đang tổng hợp câu trả lời từ dữ liệu...")

            return res
        return wrapper

    wrapped_tool_map = {name: make_logged_tool(name, fn) for name, fn in TOOL_MAP.items()}
    wrapped_tools = list(wrapped_tool_map.values())

    # Chuyển đổi lịch sử chat
    history = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    # Tắt Automatic Function Calling của SDK để tự xử lý vòng lặp tool call
    # và truyền đầy đủ function_response.id (bắt buộc với gateway/antigravity executor)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=wrapped_tools,
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # -------------------------------------------------------------------------
    # Tự động chọn danh sách model phù hợp với API key hiện tại
    # Nếu model hiện tại bị 503/404, tự động chuyển sang model tiếp theo
    # -------------------------------------------------------------------------
    models_to_try = get_models_for_key(client, api_key)
    max_retries_per_model = 3
    last_exception = None

    for model_idx, model_name in enumerate(models_to_try):
        print(f"\n🤖 [Model {model_idx + 1}/{len(models_to_try)}]: Đang thử `{model_name}`...")
        if status_cb:
            status_cb(f"Đang kết nối tới model `{model_name}`...")

        model_had_503 = False  # Theo dõi xem model này có bị 503 không

        for attempt in range(max_retries_per_model):
            try:
                executed_tools.clear()
                chart_payload = None

                contents = list(history) + [
                    types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
                ]

                response = None
                for _step in range(6):
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    fn_calls = response.function_calls
                    if not fn_calls:
                        break

                    # Giữ nguyên message chứa functionCall (bao gồm id) từ model
                    if response.candidates and response.candidates[0].content:
                        contents.append(response.candidates[0].content)

                    fn_response_parts = []
                    for fc in fn_calls:
                        fn = wrapped_tool_map.get(fc.name)
                        args = dict(fc.args) if fc.args else {}
                        res = fn(**args) if fn else {"error": f"Không tìm thấy công cụ `{fc.name}`"}
                        fn_response_parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=getattr(fc, "id", None),
                                    name=fc.name,
                                    response={"result": res},
                                )
                            )
                        )
                    contents.append(types.Content(role="user", parts=fn_response_parts))

                ans_text = response.text if (response and response.text) else "Đã phân tích xong dữ liệu."

                total_duration = time.time() - t_start
                print("-" * 60)
                model_note = f" (fallback)" if model_idx > 0 else ""
                print(f"✅ [Hoàn thành trong]: {total_duration:.2f}s | Model: {model_name}{model_note}")
                print("=" * 60 + "\n")

                return {
                    "answer": ans_text,
                    "tool_logs": executed_tools,
                    "chart": chart_payload
                }

            except Exception as e:
                last_exception = e
                err_str = str(e)
                print(f"⚠️ [{model_name} | Lần thử {attempt + 1}/{max_retries_per_model}] Lỗi: {err_str}")

                # Làm mới connection pool/socket để lần thử sau không bị dính socket hỏng
                client = get_client(api_key, force_new=True)

                # Kiểm tra xem có phải lỗi tạm thời (503/timeout) hoặc model không khả dụng (404)
                err_lower = err_str.lower()
                is_retryable = any(code in err_lower for code in [
                    "503", "unavailable", "high demand", "timeout", "timed out", "deadline", "handshake"
                ])
                is_model_unavailable = any(code in err_lower for code in [
                    "404", "not_found", "no longer available"
                ])

                if is_retryable:
                    model_had_503 = True
                    if attempt < max_retries_per_model - 1:
                        # Exponential backoff: 5s, 10s, 20s
                        wait_time = 5 * (2 ** attempt)
                        if status_cb:
                            status_cb(f"`{model_name}` đang bận (503), thử lại lần {attempt + 2}/{max_retries_per_model} sau {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Hết lần retry cho model này → chuyển sang model tiếp theo
                        break
                elif is_model_unavailable:
                    # Model không khả dụng với API key này → chuyển ngay sang model dự phòng tiếp theo
                    model_had_503 = True
                    break
                else:
                    # Lỗi không phải tạm thời (ví dụ: API key sai, lỗi cú pháp) → dừng hoàn toàn
                    print(f"❌ [LỖI NGHIÊM TRỌNG]: {err_str}")
                    return {
                        "answer": f"Đã xảy ra lỗi khi xử lý: {err_str}",
                        "tool_logs": executed_tools,
                        "chart": chart_payload
                    }

        # Nếu model hiện tại bị 503/404 hết lần retry → thông báo chuyển model
        if model_had_503 and model_idx < len(models_to_try) - 1:
            next_model = models_to_try[model_idx + 1]
            print(f"🔄 [Fallback] `{model_name}` không khả dụng/quá tải, chuyển sang `{next_model}`...")
            if status_cb:
                status_cb(f"`{model_name}` quá tải, đang chuyển sang model dự phòng `{next_model}`...")

    # Tất cả models đều thất bại
    print(f"❌ [LỖI]: Tất cả {len(models_to_try)} models đều không phản hồi: {last_exception}")
    return {
        "answer": f"⚠️ Tất cả các model AI đều đang quá tải. Vui lòng thử lại sau vài phút.\n\nChi tiết lỗi: {str(last_exception)}",
        "tool_logs": executed_tools,
        "chart": chart_payload
    }
