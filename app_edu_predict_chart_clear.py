from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# PAGE CONFIG & THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict | Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

/* Nền và kiểu chữ chung */
html, body, [class*="css"], .stApp {
    font-family: 'Be Vietnam Pro', sans-serif;
}
.stApp {
    background: #f4f7fb;
    color: #24364b;
}
.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102746 0%, #173b63 100%);
}
[data-testid="stSidebar"] * {
    color: #f4f8ff !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    color: #102746 !important;
}

/* Chữ nội dung chính */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stText, label, [data-testid="stCaptionContainer"] {
    color: #26384d;
}
h1, h2, h3, h4, h5, h6 {
    color: #173b63;
    font-weight: 700;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #26384d !important;
}

/* Tiêu đề đầu trang */
.hero {
    background: linear-gradient(115deg, #102746 0%, #1e5686 65%, #287bb0 100%);
    padding: 28px 32px;
    border-radius: 20px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 30px rgba(16,39,70,.13);
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
}
.hero p { color: #e4f0ff !important; margin: 0; font-size: .98rem; }
.eyebrow {
    color: #b9ddff;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.section-title {
    color: #163452;
    font-weight: 800;
    font-size: 1.28rem;
    margin: .3rem 0 .8rem 0;
}
.section-note {
    color: #60758c;
    font-size: .88rem;
    margin-top: -5px;
    margin-bottom: 14px;
}

/* Thẻ thông tin và khung nội dung */
.kpi {
    background: white;
    border: 1px solid #e5edf5;
    border-radius: 16px;
    padding: 17px 19px;
    box-shadow: 0 5px 18px rgba(26,54,84,.045);
    min-height: 112px;
}
.kpi-label { color: #60758c; font-size: .82rem; font-weight: 600; }
.kpi-value { color: #163452; font-size: 1.65rem; font-weight: 800; margin-top: 7px; }
.kpi-foot { color: #8293a6; font-size: .75rem; margin-top: 2px; }
.panel {
    background: white;
    border: 1px solid #e5edf5;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(26,54,84,.04);
}

/* Ô nhập liệu, danh sách chọn và bảng */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    color: #24364b !important;
    background: #ffffff !important;
}
div[data-baseweb="select"] > div {
    color: #24364b !important;
    background-color: #ffffff !important;
}
div[data-baseweb="popover"] * {
    color: #24364b !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #e5edf5;
    border-radius: 12px;
}
div[data-testid="stTabs"] button { font-weight: 700; }

/* Nút */
.stButton > button[kind="primary"],
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #1769aa, #2588c7);
    color: #ffffff !important;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    padding: .62rem 1.1rem;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover {
    background: #124f82;
    color: #ffffff !important;
}

/* Chỉ số và chân trang */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5edf5;
    padding: 14px 16px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(26,54,84,.04);
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "Nhóm3_Studentperformanceprediction.xlsx"
FEATURES = [
    "gender", "race/ethnicity", "parental level of education",
    "lunch", "test preparation course",
    "math score", "reading score", "writing score"
]
TARGET = "Grade"
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
COLORS = ["#2878B5", "#45A6A0", "#F0A44B", "#8B79C6", "#D66B78"]

# ---------------------------------------------------------
# DATA & MODEL HELPERS
# ---------------------------------------------------------
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

def make_pipeline(model_name, random_state=42):
    if model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=2500, class_weight="balanced")
    elif model_name == "Decision Tree":
        estimator = DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=4, class_weight="balanced",
            random_state=random_state
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1
        )

    categorical = FEATURES[:5]
    numeric = FEATURES[5:]
    preprocess = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric)
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])

def prepare_data(frame):
    data = frame.copy()
    for col in ["math score", "reading score", "writing score"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[TARGET] + FEATURES[5:])
    data[TARGET] = data[TARGET].astype(str)
    return data

@st.cache_data(show_spinner=False)
def compare_models(data, test_fraction, random_state):
    X = data[FEATURES].copy()
    y = data[TARGET].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=random_state, stratify=y
    )
    results, fitted = [], {}
    for name in MODEL_NAMES:
        pipe = make_pipeline(name, random_state)
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        results.append({
            "Mô hình": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0)
        })
        fitted[name] = (pipe, pred)
    return pd.DataFrame(results), y_test, fitted

def chart_layout(fig, height=340):
    # Thiết lập nền trắng và màu chữ đậm riêng cho toàn bộ biểu đồ Plotly
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=48, b=28),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Be Vietnam Pro, Arial, sans-serif",
            size=13,
            color="#172B4D"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#172B4D"),
            bgcolor="rgba(255,255,255,0.95)"
        ),
        hoverlabel=dict(
            bgcolor="#172B4D",
            font=dict(color="#FFFFFF", size=13)
        )
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="#94A3B8",
        linewidth=1,
        tickfont=dict(size=12, color="#334155"),
        title_font=dict(size=13, color="#172B4D"),
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#DCE3EC",
        gridwidth=1,
        showline=True,
        linecolor="#94A3B8",
        linewidth=1,
        tickfont=dict(size=12, color="#334155"),
        title_font=dict(size=13, color="#172B4D"),
        zeroline=False
    )
    # Đảm bảo các đường, hộp và điểm dữ liệu không bị làm trong suốt
    fig.update_traces(opacity=1)
    return fig

def kpi(label, value, foot=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 EduPredict")
    st.caption("STUDENT ANALYTICS PLATFORM")
    page = st.radio(
        "ĐIỀU HƯỚNG",
        ["Tổng quan", "So sánh mô hình", "Dự đoán xếp loại", "Dữ liệu"],
        label_visibility="visible"
    )
    st.divider()
    st.markdown("**CẤU HÌNH**")
    uploaded = st.file_uploader("Tải dữ liệu Excel", type=["xlsx"])
    test_pct = st.slider("Tỷ lệ dữ liệu kiểm tra", 10, 40, 20, 5)
    seed = st.number_input("Random state", min_value=0, max_value=9999, value=42)
    st.divider()
    st.caption("Machine Learning • Scikit-learn • Streamlit")

try:
    raw = pd.read_excel(uploaded) if uploaded is not None else load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Không thể đọc dữ liệu: {exc}")
    st.stop()

required = FEATURES + [TARGET]
missing = [c for c in required if c not in raw.columns]
if missing:
    st.error("Dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing))
    st.stop()

df = prepare_data(raw)
if df.empty or df[TARGET].nunique() < 2:
    st.error("Dữ liệu cần có ít nhất hai nhóm xếp loại và các cột điểm hợp lệ.")
    st.stop()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">EDUCATION • DATA SCIENCE • MACHINE LEARNING</div>
  <h1>Student Performance Analytics</h1>
  <p>Nền tảng phân tích dữ liệu và dự đoán kết quả học tập của sinh viên</p>
</div>
""", unsafe_allow_html=True)

grade_order = df[TARGET].value_counts().index.tolist()
avg_score = df[["math score", "reading score", "writing score"]].mean().mean()
pass_count = int((~df[TARGET].str.lower().str.contains("fail")).sum())
pass_rate = pass_count / len(df) * 100

# ---------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------
if page == "Tổng quan":
    st.markdown('<div class="section-title">Tổng quan học tập</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Bức tranh tổng thể về dữ liệu và kết quả học tập trong bộ dữ liệu.</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    with a: kpi("TỔNG SỐ SINH VIÊN", f"{len(df):,}", "Bản ghi hợp lệ")
    with b: kpi("ĐIỂM TRUNG BÌNH", f"{avg_score:.1f}/100", "Trung bình 3 môn")
    with c: kpi("TỶ LỆ KHÔNG FAIL", f"{pass_rate:.1f}%", "Theo nhãn Grade")
    with d: kpi("SỐ MỨC XẾP LOẠI", f"{df[TARGET].nunique()}", "Nhóm phân loại")

    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Phân bố xếp loại sinh viên**")
        counts = df[TARGET].value_counts().rename_axis("Xếp loại").reset_index(name="Số sinh viên")
        fig = px.pie(counts, names="Xếp loại", values="Số sinh viên", hole=.62,
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="white", width=3)))
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Điểm trung bình theo môn học**")
        scores = pd.DataFrame({
            "Môn học": ["Toán", "Đọc", "Viết"],
            "Điểm trung bình": [
                df["math score"].mean(), df["reading score"].mean(),
                df["writing score"].mean()
            ]
        })
        fig = px.bar(scores, x="Môn học", y="Điểm trung bình", text_auto=".1f",
                     color="Môn học", color_discrete_sequence=COLORS)
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_yaxes(range=[0, 100], title="")
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**So sánh điểm theo xếp loại**")
        long_scores = df.melt(id_vars=TARGET, value_vars=["math score", "reading score", "writing score"],
                              var_name="Môn", value_name="Điểm")
        long_scores["Môn"] = long_scores["Môn"].map({
            "math score": "Toán", "reading score": "Đọc", "writing score": "Viết"
        })
        fig = px.box(long_scores, x=TARGET, y="Điểm", color=TARGET,
                     color_discrete_sequence=COLORS, points=False)
        fig.update_layout(xaxis_title="", yaxis_title="Điểm số")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Cơ cấu khóa luyện thi**")
        prep = df["test preparation course"].value_counts().rename_axis("Trạng thái").reset_index(name="Số lượng")
        fig = px.bar(prep, x="Trạng thái", y="Số lượng", color="Trạng thái",
                     text_auto=True, color_discrete_sequence=["#2878B5", "#45A6A0"])
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("Lưu ý: Average Score không được dùng làm đầu vào mô hình vì đây là điểm tổng hợp từ ba môn, có thể gây rò rỉ dữ liệu mục tiêu.")

# ---------------------------------------------------------
# PAGE: MODEL COMPARISON
# ---------------------------------------------------------
elif page == "So sánh mô hình":
    st.markdown('<div class="section-title">Huấn luyện & so sánh mô hình</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Ba thuật toán được đánh giá trên cùng một tập kiểm tra để đảm bảo tính công bằng.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
    <b>Quy trình đánh giá</b><br>
    Chia dữ liệu phân tầng theo nhãn Grade → tiền xử lý dữ liệu → huấn luyện 3 mô hình →
    so sánh Accuracy, Precision, Recall và F1-score (weighted).
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("🚀 Huấn luyện và so sánh cả 3 mô hình", type="primary", use_container_width=False):
        with st.spinner("Đang huấn luyện và đánh giá các mô hình..."):
            try:
                metrics, y_test, fitted = compare_models(df, test_pct / 100, int(seed))
                st.session_state["comparison_metrics"] = metrics
                st.session_state["comparison_y_test"] = y_test
                st.session_state["comparison_fitted"] = fitted
                st.session_state["comparison_signature"] = (test_pct, int(seed), len(df))
            except Exception as exc:
                st.error(f"Không thể huấn luyện mô hình: {exc}")

    if "comparison_metrics" in st.session_state:
        metrics = st.session_state["comparison_metrics"]
        best = metrics.sort_values("F1-score", ascending=False).iloc[0]
        st.write("")
        x1, x2, x3 = st.columns(3)
        with x1: kpi("MÔ HÌNH DẪN ĐẦU", best["Mô hình"], "Theo F1-score weighted")
        with x2: kpi("F1-SCORE CAO NHẤT", f'{best["F1-score"]:.3f}', "Trên tập kiểm tra")
        with x3: kpi("ACCURACY", f'{best["Accuracy"]:.3f}', "Tỷ lệ dự đoán đúng")
        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**So sánh các chỉ số đánh giá**")
        melted = metrics.melt(id_vars="Mô hình", var_name="Chỉ số", value_name="Giá trị")
        fig = px.bar(melted, x="Mô hình", y="Giá trị", color="Chỉ số", barmode="group",
                     color_discrete_sequence=COLORS, text_auto=".3f")
        fig.update_yaxes(range=[0, 1.08], title="Điểm số")
        fig.update_xaxes(title="")
        st.plotly_chart(chart_layout(fig, 420), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown("**Bảng kết quả chi tiết**")
        display = metrics.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
            display[col] = display[col].map(lambda v: f"{v:.4f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.write("")
        chosen = st.selectbox("Xem ma trận nhầm lẫn của mô hình", MODEL_NAMES,
                              index=MODEL_NAMES.index(best["Mô hình"]))
        y_test = st.session_state["comparison_y_test"]
        model, pred = st.session_state["comparison_fitted"][chosen]
        labels = sorted(df[TARGET].astype(str).unique())
        cm = confusion_matrix(y_test, pred, labels=labels)
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                        color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(xaxis_title="Nhãn dự đoán", yaxis_title="Nhãn thực tế")
        st.plotly_chart(chart_layout(fig, 430), use_container_width=True)
        with st.expander("Xem báo cáo phân loại chi tiết"):
            report = classification_report(y_test, pred, zero_division=0, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
        st.session_state["selected_model_name"] = chosen
        st.session_state["selected_model"] = model
        st.caption("F1-score weighted là chỉ số tham khảo hữu ích khi số lượng mẫu giữa các nhóm xếp loại không đồng đều.")
    else:
        st.info("Nhấn nút **Huấn luyện và so sánh cả 3 mô hình** để xem kết quả đánh giá.")

# ---------------------------------------------------------
# PAGE: PREDICTION
# ---------------------------------------------------------
elif page == "Dự đoán xếp loại":
    st.markdown('<div class="section-title">Dự đoán kết quả học tập</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Nhập thông tin sinh viên để mô hình ước lượng nhóm xếp loại.</div>', unsafe_allow_html=True)
    if "comparison_fitted" not in st.session_state:
        st.warning("Em hãy sang trang **So sánh mô hình** và huấn luyện các mô hình trước khi dự đoán.")
    else:
        default_model = st.session_state.get("selected_model_name", "Random Forest")
        selected = st.selectbox("Chọn mô hình dự đoán", MODEL_NAMES,
                                index=MODEL_NAMES.index(default_model))
        model = st.session_state["comparison_fitted"][selected][0]
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        with st.form("student_prediction"):
            st.markdown("**Thông tin cá nhân & học tập**")
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Giới tính", sorted(df["gender"].dropna().astype(str).unique()))
                race = st.selectbox("Nhóm sắc tộc", sorted(df["race/ethnicity"].dropna().astype(str).unique()))
                parent_edu = st.selectbox("Trình độ học vấn phụ huynh",
                    sorted(df["parental level of education"].dropna().astype(str).unique()))
                lunch = st.selectbox("Loại bữa ăn", sorted(df["lunch"].dropna().astype(str).unique()))
            with c2:
                prep = st.selectbox("Khóa luyện thi",
                    sorted(df["test preparation course"].dropna().astype(str).unique()))
                math_score = st.number_input("Điểm Toán", 0.0, 100.0, 70.0, 1.0)
                reading_score = st.number_input("Điểm Đọc", 0.0, 100.0, 70.0, 1.0)
                writing_score = st.number_input("Điểm Viết", 0.0, 100.0, 70.0, 1.0)
            submitted = st.form_submit_button("🔮 Dự đoán kết quả", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if submitted:
            sample = pd.DataFrame([{
                "gender": gender, "race/ethnicity": race,
                "parental level of education": parent_edu, "lunch": lunch,
                "test preparation course": prep, "math score": math_score,
                "reading score": reading_score, "writing score": writing_score
            }])
            prediction = model.predict(sample)[0]
            st.write("")
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**KẾT QUẢ DỰ ĐOÁN**")
            st.markdown(f"<h2 style='color:#1769aa;margin:.3rem 0'>{prediction}</h2>", unsafe_allow_html=True)
            st.caption(f"Thuật toán sử dụng: {selected}")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample)[0]
                prob_df = pd.DataFrame({"Mức xếp loại": model.classes_, "Xác suất": probs})
                prob_df = prob_df.sort_values("Xác suất", ascending=True)
                fig = px.bar(prob_df, x="Xác suất", y="Mức xếp loại", orientation="h",
                             text=prob_df["Xác suất"].map(lambda v: f"{v:.1%}"),
                             color="Xác suất", color_continuous_scale="Blues")
                fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%", xaxis_title="Xác suất")
                st.plotly_chart(chart_layout(fig, 300), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Kết quả chỉ mang tính tham khảo dựa trên dữ liệu huấn luyện, không thay thế đánh giá chính thức của nhà trường.")

# ---------------------------------------------------------
# PAGE: DATA
# ---------------------------------------------------------
elif page == "Dữ liệu":
    st.markdown('<div class="section-title">Khám phá bộ dữ liệu</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Kiểm tra dữ liệu đầu vào, bộ lọc và tải dữ liệu phục vụ báo cáo.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: kpi("SỐ DÒNG", f"{len(df):,}", "Sau khi làm sạch")
    with c2: kpi("SỐ CỘT", f"{len(df.columns)}", "Bao gồm cột mục tiêu")
    with c3: kpi("GIÁ TRỊ THIẾU", f"{int(df[required].isna().sum().sum())}", "Trên các cột sử dụng")
    selected_grade = st.multiselect("Lọc theo xếp loại", sorted(df[TARGET].unique()),
                                    default=sorted(df[TARGET].unique()))
    filtered = df[df[TARGET].isin(selected_grade)]
    st.dataframe(filtered, use_container_width=True, height=430)
    st.download_button(
        "⬇️ Tải dữ liệu đã lọc (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="student_performance_filtered.csv",
        mime="text/csv"
    )

st.markdown("""
<hr style="border:0;border-top:1px solid #e4ebf2;margin-top:2.5rem">
<div style="text-align:center;color:#8a9bad;font-size:.78rem">
  EDUPREDICT • STUDENT PERFORMANCE ANALYTICS<br>
  Xây dựng với Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)