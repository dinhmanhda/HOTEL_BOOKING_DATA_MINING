# 🏨 Hotel Booking Cancellation Prediction & Data Mining

Dự án nghiên cứu và triển khai hệ thống khai phá dữ liệu nhằm dự báo rủi ro hủy phòng khách sạn bằng phương pháp **Học bán giám sát (Semi-supervised Learning)** và khai phá tri thức với **Luật kết hợp (Association Rules)**.

---

## 📋 Giới thiệu dự án
Việc hủy phòng gây tổn thất đáng kể về doanh thu và khó khăn trong quản lý vận hành khách sạn. Dự án này giải quyết bài toán bằng cách:
* Sử dụng thuật toán **Random Forest** kết hợp **Self-training** để tận dụng tối đa dữ liệu chưa có nhãn.
* Khai phá hành vi khách hàng bằng thuật toán **Apriori**.
* Đạt chỉ số **F1-Score: 0.8544** (vượt trội so với học có giám sát truyền thống 0.8267).

---

## 📂 Cấu trúc thư mục (Project Structure)



```text
BTL-DATA_MINING/
├── configs/                # Cấu hình hệ thống
├── data/                   # Quản lý tập dữ liệu
│   ├── processed/          # Dữ liệu đã qua tiền xử lý
│   └── raw/                # Dữ liệu gốc (hotel_bookings.csv)
├── notebooks/              # Tài liệu thử nghiệm (Jupyter Notebooks)
│   ├── 01_eda.ipynb        # Phân tích dữ liệu khám phá
│   ├── 02_feature_engineering.ipynb
│   ├── 03_mining.ipynb     # Khai phá luật kết hợp
│   ├── 04_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── outputs/                # Kết quả đầu ra
│   ├── figures/            # Các biểu đồ đồ họa (Ảnh 1 - 5)
│   └── models/             # Lưu trữ model (.pkl ~242MB)
├── scripts/                # Các script hỗ trợ thực thi nhanh
├── src/                    # Mã nguồn chính (Source code)
│   ├── cleaner.py          # Tiền xử lý dữ liệu
│   ├── features/           # Xây dựng đặc trưng
│   ├── mining/             # Module Apriori (association.py)
│   └── models/             # Định nghĩa Trainer và Predictor
├── venv/                   # Môi trường ảo Python
├── .gitignore              # Quản lý các file không đẩy lên Git
├── README.md               # Tài liệu hướng dẫn (File này)
└── requirements.txt        # Danh sách thư viện cài đặt 
