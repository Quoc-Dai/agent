# =============================================================================
# app.py – Giao diện web của ứng dụng, xây dựng bằng Streamlit.
# Nhiệm vụ: Hiển thị khung chat, nhận câu hỏi từ người dùng, gọi agent
# xử lý, rồi hiển thị kết quả (văn bản + tool log + biểu đồ).
# =============================================================================

import os
import pandas as pd
import plotly.express as px
import streamlit as st
from agent import run_agent  # Import hàm xử lý chính từ agent.py


# =============================================================================
# CẤU HÌNH TRANG
# Nhiệm vụ: Thiết lập tiêu đề tab trình duyệt, icon, và bố cục trang rộng.
# Phải gọi đây là lệnh đầu tiên trong file trước mọi lệnh Streamlit khác.
# =============================================================================
st.set_page_config(page_title="Sales Data AI Agent", page_icon="📊", layout="wide")
st.title("📊 AI Agent Phân Tích Dữ Liệu Bán Hàng")


# =============================================================================
# THANH BÊN (SIDEBAR)
# Nhiệm vụ: Chứa phần cấu hình (nhập API key) và gợi ý câu hỏi mẫu
# để người dùng biết có thể hỏi gì, cùng nút xóa lịch sử hội thoại.
# =============================================================================
with st.sidebar:
    st.header("Cấu hình")

    # Ô nhập API key ẩn (type="password"), tự điền nếu có biến môi trường
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))

    # Hiển thị danh sách câu hỏi gợi ý để người dùng thử nghiệm nhanh
    st.markdown("**Câu hỏi gợi ý để kiểm thử:**")
    st.markdown("- *What is the total sales value?*")
    st.markdown("- *Which five countries generated the most sales?*")
    st.markdown("- *What are the top five products by quantity sold?*")
    st.markdown("- *Show sales by month.*")
    st.markdown("- *What was our profit?*")

    # Nút xóa toàn bộ lịch sử chat, sau đó reload lại trang
    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# KHỞI TẠO SESSION STATE
# Nhiệm vụ: Tạo danh sách "messages" trong session để lưu lịch sử hội thoại
# giữa các lần rerun của Streamlit (vì Streamlit chạy lại toàn bộ script
# mỗi khi người dùng tương tác).
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =============================================================================
# HIỂN THỊ LỊCH SỬ HỘI THOẠI
# Nhiệm vụ: Vẽ lại toàn bộ tin nhắn đã có (cả user lẫn assistant) lên màn hình.
# Mỗi tin nhắn của assistant có thể kèm tool logs và biểu đồ nếu có.
# =============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Hiển thị nội dung văn bản của tin nhắn
        st.markdown(msg["content"])

        # Nếu tin nhắn có log tool calls → hiển thị trong expander có thể thu/mở
        if msg.get("tool_logs"):
            with st.expander("🛠️ Xem chi tiết Tool Calls & Kết quả tính toán"):
                for log in msg["tool_logs"]:
                    st.write(f"**Tool:** `{log['tool']}` | **Tham số:** `{log['args']}`")
                    st.json(log["result"])  # Hiển thị kết quả dạng JSON có màu sắc

        # Nếu tin nhắn có dữ liệu biểu đồ → vẽ biểu đồ tương ứng
        if msg.get("chart"):
            df_chart = pd.DataFrame(msg["chart"]["results"])
            if not df_chart.empty:
                # Chọn loại biểu đồ dựa vào "chart_type" do tool trả về
                if msg["chart"]["chart_type"] == "line":
                    fig = px.line(df_chart, x="label", y="value", markers=True, title="Biểu đồ xu hướng theo tháng")
                else:
                    fig = px.bar(df_chart, x="label", y="value", title="Biểu đồ xếp hạng")
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Ô NHẬP CÂU HỎI (CHAT INPUT)
# Nhiệm vụ: Hiển thị ô nhập liệu ở cuối trang. Khi người dùng gửi câu hỏi,
# toàn bộ khối lệnh bên trong if sẽ được thực thi.
# =============================================================================
if prompt := st.chat_input("Nhập câu hỏi về dữ liệu bán hàng của bạn..."):

    # Kiểm tra API key trước khi xử lý, nếu thiếu thì dừng và báo lỗi
    if not api_key:
        st.error("Vui lòng nhập Gemini API Key ở thanh bên trái!")
        st.stop()

    # Lưu câu hỏi của người dùng vào session và hiển thị ngay lên màn hình
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # -------------------------------------------------------------------------
    # Gọi Agent xử lý và hiển thị kết quả
    # -------------------------------------------------------------------------
    with st.chat_message("assistant"):
        # Spinner hiển thị trong lúc agent đang chạy (có thể mất vài giây)
        with st.spinner("Agent đang truy vấn dữ liệu và tính toán..."):

            # Gọi hàm run_agent với câu hỏi và lịch sử chat (trừ tin nhắn hiện tại)
            output = run_agent(prompt, st.session_state.messages[:-1], api_key)

            # Hiển thị câu trả lời dạng văn bản Markdown
            st.markdown(output["answer"])

            # Nếu có tool logs → hiển thị trong expander để debug / xem chi tiết
            if output["tool_logs"]:
                with st.expander("🛠️ Xem chi tiết Tool Calls & Kết quả tính toán"):
                    for log in output["tool_logs"]:
                        st.write(f"**Tool:** `{log['tool']}` | **Tham số:** `{log['args']}`")
                        st.json(log["result"])

            # Nếu có dữ liệu biểu đồ → vẽ biểu đồ tương ứng bên dưới câu trả lời
            if output["chart"]:
                df_chart = pd.DataFrame(output["chart"]["results"])
                if not df_chart.empty:
                    if output["chart"]["chart_type"] == "line":
                        fig = px.line(df_chart, x="label", y="value", markers=True, title="Biểu đồ xu hướng theo tháng")
                    else:
                        fig = px.bar(df_chart, x="label", y="value", title="Biểu đồ xếp hạng")
                    st.plotly_chart(fig, use_container_width=True)

            # Lưu câu trả lời của assistant vào session để hiển thị lại khi rerun
            st.session_state.messages.append({
                "role": "assistant",
                "content": output["answer"],
                "tool_logs": output["tool_logs"],
                "chart": output["chart"]
            })
