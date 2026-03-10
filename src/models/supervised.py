from sklearn.ensemble import RandomForestClassifier

def train_baseline(X, y):
    """Huấn luyện mô hình cơ bản để so sánh."""
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model