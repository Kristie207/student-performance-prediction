from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction")
st.caption("Ứng dụng phân tích dữ liệu và dự đoán xếp loại học tập của sinh viên")

DEFAULT_FILE = Path(__file__).parent / "Nhóm3_Studentperformanceprediction.xlsx"

@st.cache_data
def load_excel(file_or_path):
    return pd.read_excel(file_or_path)

def train_and_evaluate(data, model_name, test_size, random_state):
    feature_cols = [
        "gender", "race/ethnicity", "parental level of education",
        "lunch", "test preparation course",
        "math score", "reading score", "writing score"
    ]
    target_col = "Grade"
    X = data[feature_cols].copy()
    y = data[target_col].astype(str)

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    transformers = []
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    preprocessor = ColumnTransformer(transformers)

    if model_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "Decision Tree":
        classifier = DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=4, class_weight="balanced",
            random_state=random_state
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    evaluation_model = Pipeline([
        ("preprocessor", preprocessor), ("classifier", classifier)
    ])
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (weighted)": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }
    final_model = Pipeline([
        ("preprocessor", preprocessor), ("classifier", classifier)
    ])
    final_model.fit(X, y)
    return final_model, metrics, y_test, predictions

with st.sidebar:
    st.header("⚙️ Thiết lập dữ liệu")
    uploaded = st.file_uploader("Tải lên file Excel (.xlsx)", type=["xlsx"])
    test_size_pct = st.slider("Tỷ lệ dữ liệu kiểm tra", 10, 40, 20, 5)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    st.divider()
    st.caption("Nếu không tải file mới, ứng dụng sẽ sử dụng file Excel nằm cùng thư mục với app.py.")

try:
    if uploaded is not None:
        df = load_excel(uploaded)
    elif DEFAULT_FILE.exists():
        df = load_excel(DEFAULT_FILE)
    else:
        st.warning("Chưa tìm thấy file Excel. Hãy tải file dữ liệu lên bằng ô tải tệp ở thanh bên trái.")
        st.stop()
except Exception as exc:
    st.error(f"Không thể đọc file Excel: {exc}")
    st.stop()

required_cols = [
    "gender", "race/ethnicity", "parental level of education",
    "lunch", "test preparation course", "math score",
    "reading score", "writing score", "Grade"
]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error("File dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing))
    st.stop()

df = df.dropna(subset=["Grade"]).copy()
for col in ["math score", "reading score", "writing score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["math score", "reading score", "writing score"])
if df["Grade"].nunique() < 2:
    st.error("Cần ít nhất hai nhóm Grade để huấn luyện mô hình.")
    st.stop()

st.success(f"Đã tải dữ liệu thành công: {len(df):,} dòng dữ liệu.")
tab_overview, tab_model, tab_predict = st.tabs([
    "📊 Tổng quan dữ liệu", "🧠 Huấn luyện & đánh giá", "🔮 Dự đoán xếp loại"
])

with tab_overview:
    st.subheader("Tổng quan bộ dữ liệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số sinh viên", f"{len(df):,}")
    c2.metric("Số thuộc tính", len(required_cols) - 1)
    c3.metric("Số mức xếp loại", df["Grade"].nunique())
    c4.metric("Dữ liệu thiếu", f"{int(df[required_cols].isna().sum().sum())}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Phân bố xếp loại**")
        grade_counts = df["Grade"].value_counts().rename_axis("Grade").reset_index(name="Số lượng")
        st.bar_chart(grade_counts.set_index("Grade"))
    with right:
        st.markdown("**Điểm trung bình theo môn**")
        st.bar_chart(df[["math score", "reading score", "writing score"]].mean())
    st.markdown("**Bảng dữ liệu**")
    st.dataframe(df, use_container_width=True, height=350)
    st.download_button(
        "⬇️ Tải dữ liệu đang sử dụng (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="student_performance_data.csv", mime="text/csv"
    )

with tab_model:
    st.subheader("Huấn luyện và đánh giá mô hình")
    model_name = st.selectbox(
        "Chọn thuật toán",
        ["Random Forest", "Logistic Regression", "Decision Tree"]
    )
    st.caption(
        f"Dữ liệu được chia thành {100-test_size_pct}% huấn luyện và "
        f"{test_size_pct}% kiểm tra. Các chỉ số được tính trên tập kiểm tra."
    )
    if st.button("🚀 Huấn luyện mô hình", type="primary"):
        with st.spinner("Đang huấn luyện và đánh giá..."):
            try:
                model, metrics, y_test, y_pred = train_and_evaluate(
                    df, model_name, test_size_pct / 100, int(random_state)
                )
                st.session_state["trained_model"] = model
                st.session_state["trained_model_name"] = model_name
                st.session_state["metrics"] = metrics
                st.session_state["y_test"] = y_test
                st.session_state["y_pred"] = y_pred
            except Exception as exc:
                st.error(f"Không thể huấn luyện mô hình: {exc}")

    if "metrics" in st.session_state:
        st.markdown(f"#### Kết quả: {st.session_state['trained_model_name']}")
        metric_cols = st.columns(4)
        for col, (label, value) in zip(metric_cols, st.session_state["metrics"].items()):
            col.metric(label, f"{value:.3f}")
        y_test, y_pred = st.session_state["y_test"], st.session_state["y_pred"]
        labels = sorted(df["Grade"].astype(str).unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
            ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False
        )
        ax.set_title("Ma trận nhầm lẫn")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        st.markdown("**Báo cáo phân loại chi tiết**")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
        st.info(
            "Accuracy thể hiện tỷ lệ dự đoán đúng tổng thể. Precision, Recall và "
            "F1-score weighted phản ánh hiệu quả trên các lớp có xét đến số lượng mẫu. "
            "Nên xem đồng thời các chỉ số và ma trận nhầm lẫn."
        )
    else:
        st.info("Chọn thuật toán và nhấn **Huấn luyện mô hình** để xem kết quả.")

with tab_predict:
    st.subheader("Nhập thông tin để dự đoán xếp loại")
    st.write("Điền thông tin của một sinh viên. Mô hình sẽ dự đoán nhóm xếp loại học tập.")
    if "trained_model" not in st.session_state:
        st.warning("Hãy huấn luyện mô hình ở tab **Huấn luyện & đánh giá** trước khi dự đoán.")
    else:
        with st.form("prediction_form"):
            a, b = st.columns(2)
            with a:
                gender = st.selectbox("Giới tính", sorted(df["gender"].dropna().astype(str).unique()))
                race = st.selectbox("Nhóm chủng tộc/sắc tộc", sorted(df["race/ethnicity"].dropna().astype(str).unique()))
                parent_edu = st.selectbox(
                    "Trình độ học vấn của phụ huynh",
                    sorted(df["parental level of education"].dropna().astype(str).unique())
                )
                lunch = st.selectbox("Loại bữa ăn", sorted(df["lunch"].dropna().astype(str).unique()))
            with b:
                prep = st.selectbox(
                    "Khóa luyện thi",
                    sorted(df["test preparation course"].dropna().astype(str).unique())
                )
                math_score = st.number_input("Điểm Toán", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
                reading_score = st.number_input("Điểm Đọc", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
                writing_score = st.number_input("Điểm Viết", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            submitted = st.form_submit_button("🔍 Dự đoán kết quả", type="primary")
        if submitted:
            sample = pd.DataFrame([{
                "gender": gender, "race/ethnicity": race,
                "parental level of education": parent_edu, "lunch": lunch,
                "test preparation course": prep, "math score": math_score,
                "reading score": reading_score, "writing score": writing_score
            }])
            model = st.session_state["trained_model"]
            prediction = model.predict(sample)[0]
            st.success(f"### Kết quả dự đoán: {prediction}")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                probability_df = pd.DataFrame({
                    "Mức xếp loại": model.classes_, "Xác suất dự đoán": probabilities
                }).sort_values("Xác suất dự đoán", ascending=False)
                st.markdown("**Xác suất theo từng mức xếp loại**")
                display_df = probability_df.copy()
                display_df["Xác suất dự đoán"] = display_df["Xác suất dự đoán"].map(lambda x: f"{x:.1%}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.bar_chart(probability_df.set_index("Mức xếp loại"))
            st.caption(
                "Kết quả chỉ mang tính tham khảo dựa trên dữ liệu và mô hình đã huấn luyện, "
                "không thay thế đánh giá chính thức của nhà trường."
            )

st.divider()
st.caption("Student Performance Prediction • Streamlit + Scikit-learn")
