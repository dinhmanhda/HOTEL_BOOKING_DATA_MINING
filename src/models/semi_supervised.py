import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier

# Đảm bảo phần mở ngoặc có tham số labeled_ratio
def run_self_training(model, X, y, labeled_ratio=0.2):
    rng = np.random.RandomState(42)
    # Tạo mặt nạ dựa trên tỷ lệ labeled_ratio
    unlabeled_mask = rng.rand(len(y)) > labeled_ratio
    
    y_train = y.copy()
    y_train[unlabeled_mask] = -1 
    
    self_training_model = SelfTrainingClassifier(model)
    self_training_model.fit(X, y_train)
    return self_training_model