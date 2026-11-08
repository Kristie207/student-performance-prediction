# EduPredict – Student Performance Analytics

Ứng dụng Streamlit hỗ trợ khám phá dữ liệu, so sánh thuật toán Machine Learning và dự đoán xếp loại học tập của sinh viên.

## Điểm nổi bật
- **Dashboard tổng quan:** KPI, phân bố xếp loại, điểm trung bình theo môn, điểm theo nhóm xếp loại và tình trạng luyện thi.
- **So sánh mô hình:** Logistic Regression, Decision Tree và Random Forest trên cùng tập kiểm tra.
- **Đánh giá mô hình:** Accuracy, Precision, Recall, F1-score weighted, confusion matrix và classification report.
- **Dự đoán tương tác:** nhập thông tin học tập, chọn mô hình và xem xếp loại dự đoán cùng xác suất từng lớp.
- **Khám phá dữ liệu:** lọc theo xếp loại, xem bảng dữ liệu và tải CSV.
- **Tải dữ liệu Excel:** có thể sử dụng file mặc định trong repository hoặc tải một file khác qua sidebar.

## Cấu trúc dự án
```
student-performance-prediction/
├── app.py
├── requirements.txt
├── README.md
└── Nhóm3_Studentperformanceprediction.xlsx
```

## Chạy ứng dụng trên máy tính

Yêu cầu Python 3.10 trở lên.

1. Clone repository:
   ```bash
   git clone https://github.com/Kristie207/student-performance-prediction.git
   cd student-performance-prediction
   ```
2. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```
3. Khởi chạy:
   ```bash
   streamlit run app.py
   ```

## Dữ liệu đầu vào

File Excel cần có các cột:
- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `math score`
- `reading score`
- `writing score`
- `Grade`

Ứng dụng không sử dụng `Average Score` làm đặc trưng đầu vào vì biến này được tính từ điểm ba môn, có thể gây rò rỉ dữ liệu mục tiêu.

## Triển khai trên Streamlit Community Cloud

1. Truy cập https://share.streamlit.io/ và đăng nhập bằng GitHub.
2. Chọn **Create app**.
3. Chọn repository `Kristie207/student-performance-prediction`, branch `main`, main file path `app.py`.
4. Chọn **Deploy**. Nếu ứng dụng đã được triển khai, Streamlit Cloud sẽ tự cập nhật sau khi nhận commit mới.
5. Kiểm tra file Excel nằm ở thư mục gốc repository để ứng dụng tự nạp dữ liệu.

## Lưu ý về đánh giá

Các mô hình sử dụng cùng một cách chia tập train/test có phân tầng theo nhãn để so sánh công bằng hơn. F1-score weighted được trình bày cùng Accuracy, Precision và Recall nhằm phản ánh kết quả trong trường hợp số lượng mẫu giữa các nhóm xếp loại không cân bằng. Kết quả dự đoán chỉ mang tính tham khảo.
