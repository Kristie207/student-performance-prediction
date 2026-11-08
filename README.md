# Student Performance Prediction

Ứng dụng Streamlit phân tích dữ liệu và dự đoán xếp loại học tập bằng Machine Learning.

## Tính năng
- Tổng quan dữ liệu, phân bố xếp loại và điểm trung bình theo môn.
- Huấn luyện và đánh giá Logistic Regression, Decision Tree và Random Forest.
- Hiển thị Accuracy, Precision, Recall, F1-score, confusion matrix và classification report.
- Nhập thông tin sinh viên để dự đoán Grade và xác suất theo từng nhóm.
- Tải dữ liệu đang sử dụng về định dạng CSV.

## Cấu trúc dự án
```
student-performance-prediction/
├── app.py
├── requirements.txt
├── README.md
└── Nhóm3_Studentperformanceprediction.xlsx  # thêm file dữ liệu này để tự nạp dữ liệu
```

## Chạy ứng dụng trên máy tính
Yêu cầu Python 3.10 trở lên.

1. Tải hoặc clone repository:
   ```bash
   git clone https://github.com/Kristie207/student-performance-prediction.git
   cd student-performance-prediction
   ```
2. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy ứng dụng:
   ```bash
   streamlit run app.py
   ```

## Dữ liệu đầu vào
Đưa file `Nhóm3_Studentperformanceprediction.xlsx` vào cùng thư mục với `app.py`. Nếu chưa thêm file này, ứng dụng vẫn chạy nhưng cần tải file Excel bằng ô tải tệp ở thanh bên trái.

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

Không sử dụng `Average Score` làm đặc trưng đầu vào vì biến này được suy ra từ điểm các môn, có thể gây rò rỉ dữ liệu.

## Triển khai miễn phí bằng Streamlit Community Cloud
1. Đăng nhập tại https://share.streamlit.io/ bằng tài khoản GitHub.
2. Chọn **Create app**.
3. Chọn repository `Kristie207/student-performance-prediction`, branch `main`, main file path `app.py`.
4. Nhấn **Deploy**.
5. Đảm bảo file Excel đã được thêm vào repository để ứng dụng tự nạp dữ liệu. Nếu chưa thêm, có thể tải file Excel trực tiếp trong thanh bên của ứng dụng.

## Lưu ý
Các chỉ số được tính trên tập kiểm tra tách riêng. Kết quả dự đoán chỉ mang tính tham khảo, không thay thế đánh giá chính thức của nhà trường.
