from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def train_hotel_model(df):
    # 1. Chọn đặc trưng (X) và mục tiêu (y)
    # Loại bỏ 'is_canceled' vì đó là cái cần dự đoán
    X = pd.get_dummies(df.drop(columns=['is_canceled']))
    y = df['is_canceled']

    # 2. Chia tập dữ liệu: 80% để học, 20% để kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Khởi tạo và huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
    print("\nBáo cáo chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred))
    
    return model