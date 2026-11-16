# =============================================================================
# agent.py – "Bộ não" của AI Agent: quản lý vòng lặp suy luận và tool calling.
# Flow: nhận câu hỏi → gọi Gemini API → model tự quyết định tool nào cần dùng
#       → thực thi tool → trả kết quả về model → lặp lại cho đến khi có câu trả lời.
# =============================================================================

import os
from google import genai
from google.genai import types

# Import tất cả các tool đã định nghĩa trong tools.py
from tools import (
    inspect_dataset,
    get_total_sales,
    get_top_rankings,
    get_sales_by_month,
)


# =============================================================================
# SYSTEM PROMPT – Hướng dẫn hành vi cho AI
# Nhiệm vụ: Quy định AI phải làm gì, không được làm gì, và cách xử lý
# các tình huống đặc biệt (thiếu dữ liệu, kết quả rỗng, v.v.)
# =============================================================================
SYSTEM_PROMPT = """
Bạn là một AI Data Analyst chuyên phân tích dữ liệu bán hàng từ cơ sở dữ liệu.
Hãy tuân thủ nghiêm ngặt các quy tắc sau:
1. LUÔN gọi tool `inspect_dataset` đầu tiên nếu chưa biết cấu trúc bảng.
2. Mọi con số đưa ra trong câu trả lời BẮT BUỘC phải lấy từ kết quả thực thi của các tools. Tuyệt đối không tự tính nhẩm hay bịa số liệu.
3. Xử lý thiếu dữ liệu (Data Limitations): Nếu người dùng hỏi thông tin không có trong bảng (ví dụ: lợi nhuận / profit, giá vốn / cost), hãy giải thích rõ rằng bộ dữ liệu chỉ có đơn giá bán (UnitPrice) và số lượng (Quantity) mà không có dữ liệu chi phí/giá vốn nên không thể tính toán.
4. Luôn nêu rõ giả định (Assumptions) ở cuối câu trả lời về việc đã loại trừ các đơn hủy hoặc giao dịch có Quantity <= 0, UnitPrice <= 0.
5. Nếu kết quả từ tool rỗng (empty), hãy thông báo rõ ràng cho người dùng.
"""


# =============================================================================
# TOOL MAP – Ánh xạ tên tool (chuỗi) → hàm Python tương ứng
# Nhiệm vụ: Khi model trả về yêu cầu gọi tool theo tên, dùng dict này
# để tìm đúng hàm cần thực thi mà không cần if/elif dài dòng.
# =============================================================================
TOOL_MAP = {
    "inspect_dataset": inspect_dataset,
    "get_total_sales": get_total_sales,
    "get_top_rankings": get_top_rankings,
    "get_sales_by_month": get_sales_by_month,
}


# =============================================================================
# HÀM CHÍNH – run_agent
# Nhiệm vụ: Điều phối toàn bộ quá trình xử lý một câu hỏi của người dùng.
# Input  : câu hỏi (user_message), lịch sử hội thoại (chat_history), API key
# Output : dict gồm { answer, tool_logs, chart }
# =============================================================================
def run_agent(user_message: str, chat_history: list, api_key: str) -> dict:
    """Chạy vòng lặp Agent, trả về câu trả lời, log tool calls và dữ liệu biểu đồ (nếu có)."""

    # Khởi tạo Gemini client với API key được truyền vào từ giao diện
    client = genai.Client(api_key=api_key)

    # ---------------------------------------------------------------------
    # Chuyển đổi lịch sử hội thoại sang định dạng Gemini API hiểu được
    # (mỗi tin nhắn là một Content object với role "user" hoặc "model")
    # ---------------------------------------------------------------------
    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    # Thêm câu hỏi mới nhất của người dùng vào cuối danh sách
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    # Danh sách ghi lại tất cả tool nào đã được gọi và kết quả trả về
    tool_logs = []

    # Dữ liệu biểu đồ (nếu tool nào trả về chart_type thì lưu vào đây)
    chart_payload = None

    # ---------------------------------------------------------------------
    # Cấu hình cho model Gemini:
    # - system_instruction : vai trò và quy tắc của AI
    # - tools              : danh sách hàm Python model có thể yêu cầu gọi
    # - automatic_function_calling disabled : agent tự xử lý tool, không để SDK tự gọi
    # - temperature=0.1    : giảm sự sáng tạo, ưu tiên trả lời chính xác, nhất quán
    # ---------------------------------------------------------------------
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=list(TOOL_MAP.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.1,
    )

    # =================================================================
    # VÒNG LẶP TOOL CALLING (tối đa 5 bước suy luận)
    # Mỗi vòng lặp: gọi Gemini → kiểm tra xem model có muốn dùng tool
    # không → nếu có thì thực thi tool → đưa kết quả trở lại model
    # → model tiếp tục suy luận → lặp lại cho đến khi có câu trả lời
    # =================================================================
    for _ in range(5):

        # Gửi toàn bộ hội thoại + lịch sử tool cho Gemini để suy luận
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=contents,
            config=config,
        )

        # Lấy phản hồi từ model và thêm vào chuỗi hội thoại
        candidate = response.candidates[0]
        contents.append(candidate.content)

        # Kiểm tra model có yêu cầu gọi tool nào không
        function_calls = [
            part.function_call for part in candidate.content.parts if part.function_call
        ]

        # Nếu không còn tool nào cần gọi → model đã có câu trả lời cuối
        if not function_calls:
            return {
                "answer": response.text,       # Văn bản trả lời cho người dùng
                "tool_logs": tool_logs,         # Danh sách tool đã được gọi
                "chart": chart_payload          # Dữ liệu biểu đồ (nếu có)
            }

        # -------------------------------------------------------------
        # Thực thi từng tool mà model yêu cầu
        # -------------------------------------------------------------
        tool_response_parts = []
        for fc in function_calls:
            func_name = fc.name                            # Tên tool (chuỗi)
            func_args = dict(fc.args) if fc.args else {}  # Tham số truyền vào tool

            # Tìm hàm Python tương ứng trong TOOL_MAP và thực thi
            func = TOOL_MAP.get(func_name)
            result = func(**func_args) if func else {"error": f"Tool {func_name} không tồn tại"}

            # Ghi lại thông tin tool call để hiển thị trên giao diện
            tool_logs.append({
                "tool": func_name,
                "args": func_args,
                "result": result
            })

            # Nếu tool trả về dữ liệu biểu đồ thì lưu lại để render
            if isinstance(result, dict) and "chart_type" in result and "results" in result:
                chart_payload = result

            # Đóng gói kết quả tool thành định dạng Gemini API yêu cầu
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=func_name,
                    response={"result": result}
                )
            )

        # Thêm kết quả của tất cả tool vào hội thoại để model đọc tiếp
        contents.append(types.Content(role="user", parts=tool_response_parts))

    # Nếu sau 5 vòng lặp vẫn chưa có câu trả lời → thông báo vượt giới hạn
    return {
        "answer": "Đã vượt quá số bước xử lý tối đa.",
        "tool_logs": tool_logs,
        "chart": chart_payload
    }
