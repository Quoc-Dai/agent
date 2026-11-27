# =============================================================================
# app.py – Giao diện web của ứng dụng, xây dựng bằng Streamlit.
# Nhiệm vụ: Hiển thị khung chat, dashboard tổng quan, nhận câu hỏi từ người
# dùng, gọi agent xử lý, rồi hiển thị kết quả (văn bản + tool log + biểu đồ).
# =============================================================================

import os
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai

from agent import DEFAULT_FALLBACK_MODELS, GEMINI_BASE_URL, get_client, run_agent
from tools import DB_PATH, get_data_info, get_total_sales, init_database, reset_database

# Nạp biến môi trường từ file .env (nếu có)
load_dotenv()


# =============================================================================
# CẤU HÌNH TRANG (Phải là lệnh Streamlit đầu tiên)
# =============================================================================
st.set_page_config(
    page_title="Sales Data AI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS TÙY CHỈNH – Giao diện hiện đại, tương thích cả Light & Dark Mode
# =============================================================================
st.markdown(
    """
    <style>
    /* Thu gọn khoảng trắng thừa trên cùng */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1120px;
    }

    /* Hero Banner đầu trang */
    .hero-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: #ffffff;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        opacity: 0.92;
        margin: 0;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.9rem;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    }
    .kpi-label {
        font-size: 0.8rem;
        opacity: 0.72;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        line-height: 1.2;
    }

    /* Sidebar Data Card */
    .sidebar-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ok {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Nút gợi ý câu hỏi */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        border-radius: 12px;
        text-align: left;
        transition: all 0.15s ease;
    }

    /* Thời gian xử lý badge */
    .latency-badge {
        display: inline-block;
        font-size: 0.78rem;
        opacity: 0.75;
        margin-top: 0.25rem;
    }

    /* Che API Key bằng CSS thay cho type="password" để trình duyệt không bật popup gợi ý mật khẩu */
    input[aria-label="Gemini API Key"] {
        -webkit-text-security: disc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# KHỞI TẠO SESSION STATE
# =============================================================================
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = bool(st.session_state.api_key)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# =============================================================================
# CACHE KPI TỔNG QUAN (Chỉ chạy query nhẹ khi database đã tồn tại)
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_overview_kpis():  # Lấy và cache các chỉ số KPI tổng quan (doanh thu, số giao dịch, số lượng) trong 300s
    """Truy vấn và cache các chỉ số KPI tổng quan (doanh thu, số giao dịch, tổng số lượng) trong 300 giây."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        return get_total_sales(include_cancellations=False)
    except Exception:
        return None


# =============================================================================
# THANH BÊN (SIDEBAR)
# =============================================================================
with st.sidebar:
    st.markdown("### 🔑 Cấu hình API Key")

    input_key = st.text_input(
        "Gemini API Key",
        type="default",
        autocomplete="off",
        value=st.session_state.api_key,
        placeholder="AIzaSy...",
        help="Lấy API Key miễn phí tại: https://aistudio.google.com/apikey",
    )

    col_btn, col_status = st.columns([1.15, 1])
    with col_btn:
        if st.button("Xác nhận Key", type="primary", use_container_width=True):
            if not input_key.strip():
                st.session_state.api_key_valid = False
                st.error("Vui lòng không để trống API Key!")
            else:
                with st.spinner("Đang xác thực..."):
                    try:
                        test_client = get_client(input_key.strip(), force_new=True)
                        if GEMINI_BASE_URL:
                            # Gọi trực tiếp generateContent tới gateway cấu hình (v1beta/models/gemini-3-flash:generateContent)
                            test_client.models.generate_content(
                                model=DEFAULT_FALLBACK_MODELS[0],
                                contents="hi",
                            )
                        else:
                            next(iter(test_client.models.list()), None)

                        st.session_state.api_key = input_key.strip()
                        st.session_state.api_key_valid = True
                        st.toast("✅ Xác thực API Key thành công!", icon="🎉")
                    except Exception as e:
                        st.session_state.api_key_valid = False
                        st.error(f"❌ Key không hợp lệ: {str(e)[:100]}")

    with col_status:
        if st.session_state.api_key_valid:
            st.markdown('<div class="status-pill status-ok">● Đã kích hoạt</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill status-warn">○ Chưa xác nhận</div>', unsafe_allow_html=True)

    st.divider()

    # =========================================================================
    # QUẢN LÝ DỮ LIỆU
    # =========================================================================
    st.markdown("### 📁 Quản lý dữ liệu")

    data_info = get_data_info()
    if data_info:
        st.markdown(
            f"""
            <div class="sidebar-card">
                📄 <b>Nguồn:</b> <code>{data_info['source_file']}</code><br>
                📊 <b>Số dòng:</b> {data_info['row_count']:,}<br>
                🕐 <b>Cập nhật:</b> {data_info['built_at']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Chưa có dữ liệu. Hãy đặt file Excel/CSV vào thư mục dự án.")

    if st.button("🔄 Tải lại dữ liệu", use_container_width=True):
        with st.spinner("Đang rebuild database..."):
            try:
                reset_database()
                init_database()
                st.cache_data.clear()
                st.session_state.messages = []
                st.toast("✅ Đã tải lại dữ liệu!", icon="🔄")
                time.sleep(0.8)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)[:200]}")

    st.divider()

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# HÀM HIỂN THỊ BIỂU ĐỒ (CHART RENDERER)
# =============================================================================
def render_chart(chart_info: dict, key: str = None):  # Vẽ biểu đồ Plotly (area/line theo tháng hoặc bar xếp hạng) lên giao diện Streamlit
    """Vẽ biểu đồ tương tác Plotly (biểu đồ vùng/đường cho chuỗi thời gian hoặc biểu đồ cột cho xếp hạng).

    Args:
        chart_info: Dict kết quả từ tool có chứa `results` và `chart_type` (`line` hoặc `bar`).
        key: Khóa định danh duy nhất cho component `st.plotly_chart` trong Streamlit.
    """
    if not chart_info or not chart_info.get("results"):
        return
    df_chart = pd.DataFrame(chart_info["results"])
    if df_chart.empty:
        return

    if chart_info.get("chart_type") == "line":
        fig = px.area(
            df_chart,
            x="label",
            y="value",
            markers=True,
            title="📈 Xu hướng doanh thu theo tháng (USD)",
            labels={"label": "Tháng", "value": "Doanh thu (USD)"},
        )
        fig.update_traces(
            line=dict(color="#4f46e5", width=3),
            marker=dict(size=7, color="#0ea5e9", line=dict(width=2, color="#ffffff")),
            fillcolor="rgba(79, 70, 229, 0.14)",
            hovertemplate="<b>%{x}</b><br>Doanh thu: $%{y:,.2f}<extra></extra>",
        )
    else:
        is_sales = chart_info.get("metric") == "total_sales_value" or chart_info.get("currency") == "USD"
        y_label = "Doanh thu (USD)" if is_sales else "Số lượng bán"
        group_col = chart_info.get("group_by", "Danh mục")
        title = f"📊 Xếp hạng theo {y_label}"

        fig = px.bar(
            df_chart,
            x="label",
            y="value",
            title=title,
            labels={"label": group_col, "value": y_label},
            color="value",
            color_continuous_scale=["#38bdf8", "#6366f1", "#4338ca"],
        )
        value_fmt = "$%{y:,.2f}" if is_sales else "%{y:,.0f}"
        fig.update_traces(
            marker_line_width=0,
            hovertemplate=f"<b>%{{x}}</b><br>{y_label}: {value_fmt}<extra></extra>",
        )
        fig.update_layout(coloraxis_showscale=False)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        title_font=dict(size=16),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)", zeroline=False),
        hoverlabel=dict(bgcolor="#1e293b", font_color="#ffffff", font_size=13),
    )
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


# =============================================================================
# HERO BANNER & KPI TỔNG QUAN
# =============================================================================
key_badge = "🔑 API Key: Sẵn sàng" if st.session_state.api_key_valid else "🔑 API Key: Chưa xác nhận"
db_badge = f"🗄️ Dữ liệu: {data_info['source_file']}" if data_info else "🗄️ Dữ liệu: Chưa nạp"

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">📊 Sales Data AI Agent</div>
        <div class="hero-subtitle">
            Trợ lý phân tích dữ liệu bán hàng thông minh — Hỏi bằng ngôn ngữ tự nhiên, nhận số liệu & biểu đồ tức thì.
        </div>
        <div class="hero-badges">
            <span class="hero-badge">{key_badge}</span>
            <span class="hero-badge">{db_badge}</span>
            <span class="hero-badge">⚡ Powered by Gemini & DuckDB</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hiển thị KPI cards nếu database đã sẵn sàng
kpis = load_overview_kpis()
if kpis and kpis.get("total_sales_value") is not None:
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">💰 TỔNG DOANH THU HỢP LỆ</div>
                <div class="kpi-value">${kpis['total_sales_value']:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">🧾 SỐ GIAO DỊCH HỢP LỆ</div>
                <div class="kpi-value">{kpis['valid_transactions']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">📦 TỔNG SỐ LƯỢNG BÁN RA</div>
                <div class="kpi-value">{int(kpis['total_units_sold']):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


# =============================================================================
# MÀN HÌNH CHÀO MỪNG & CÂU HỎI GỢI Ý (Khi chưa có hội thoại)
# =============================================================================
SUGGESTIONS = [
    ("💰 Tổng doanh thu", "What is the total sales value?"),
    ("🌍 Top 5 quốc gia", "Which five countries generated the most sales?"),
    ("🏆 Top 5 sản phẩm", "What are the top five products by quantity sold?"),
    ("📈 Doanh thu theo tháng", "Show sales by month."),
    ("📅 Doanh thu theo năm", "Show sales by year."),
    ("💡 Hỏi về lợi nhuận", "What was our profit?"),
]

if not st.session_state.messages:
    st.markdown("#### ✨ Gợi ý câu hỏi nhanh")
    cols = st.columns(3)
    for idx, (label, query) in enumerate(SUGGESTIONS):
        with cols[idx % 3]:
            if st.button(f"{label}\n\n`{query}`", key=f"suggest_{idx}", use_container_width=True):
                st.session_state.pending_prompt = query
                st.rerun()


# =============================================================================
# HIỂN THỊ LỊCH SỬ HỘI THOẠI
# =============================================================================
for idx, msg in enumerate(st.session_state.messages):
    avatar = "🧑‍💼" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg.get("duration") is not None:
            st.markdown(f'<div class="latency-badge">⚡ Xử lý trong {msg["duration"]:.2f}s</div>', unsafe_allow_html=True)

        if msg.get("tool_logs"):
            with st.expander("🛠️ Xem chi tiết Tool Calls & Kết quả tính toán"):
                for log in msg["tool_logs"]:
                    st.write(f"**Tool:** `{log['tool']}` | **Tham số:** `{log['args']}`")
                    st.json(log["result"])

        if msg.get("chart"):
            render_chart(msg["chart"], key=f"chart_history_{idx}")


# =============================================================================
# NHẬN CÂU HỎI TỪ CHAT INPUT HOẶC NÚT GỢI Ý
# =============================================================================
chat_prompt = st.chat_input("Nhập câu hỏi về dữ liệu bán hàng của bạn...")
prompt = chat_prompt or st.session_state.pending_prompt
if st.session_state.pending_prompt:
    st.session_state.pending_prompt = None

if prompt:
    active_key = st.session_state.get("api_key", "").strip()
    if not active_key or not st.session_state.get("api_key_valid", False):
        st.error("⚠️ Vui lòng nhập và bấm nút **'Xác nhận Key'** ở thanh bên trái trước khi bắt đầu chat!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        status_box = st.status("🚀 Agent đang tiếp nhận câu hỏi...", expanded=True)

        def update_status(text):  # Cập nhật dòng trạng thái xử lý realtime trong st.status khi Agent chạy
            """Cập nhật nhãn trạng thái đang chạy trong `st.status` khi Agent chuyển bước."""
            status_box.update(label=f"⚡ {text}", state="running")

        t_req_start = time.time()
        output = run_agent(prompt, st.session_state.messages[:-1], active_key, status_cb=update_status)
        req_duration = time.time() - t_req_start
        status_box.update(label=f"✅ Hoàn tất trong {req_duration:.2f}s!", state="complete", expanded=False)

        full_answer = output.get("answer", "")
        st.markdown(full_answer)
        st.markdown(f'<div class="latency-badge">⚡ Xử lý trong {req_duration:.2f}s</div>', unsafe_allow_html=True)

        if output["tool_logs"]:
            with st.expander("🛠️ Xem chi tiết Tool Calls & Kết quả tính toán"):
                for log in output["tool_logs"]:
                    st.write(f"**Tool:** `{log['tool']}` | **Tham số:** `{log['args']}`")
                    st.json(log["result"])

        if output["chart"]:
            render_chart(output["chart"], key=f"chart_current_{time.time()}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "tool_logs": output["tool_logs"],
            "chart": output["chart"],
            "duration": req_duration,
        })
