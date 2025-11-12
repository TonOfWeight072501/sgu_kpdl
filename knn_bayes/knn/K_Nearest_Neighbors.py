import numpy as np
from collections import Counter

class K_Nearest_Neighbors_fix:
    """
    Phiên bản KNN này có thể tương thích với các công cụ như K-Fold.
    """

    def __init__(self, k=5):
        """Khởi tạo mô hình CHỈ với siêu tham số K."""
        self.k = k
        # Dữ liệu huấn luyện sẽ được lưu khi gọi .fit()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        "Học" (lưu trữ) dữ liệu huấn luyện.
        KNN là "lazy learner", nên hàm fit chỉ đơn giản là lưu trữ dữ liệu.
        
        Tham số:
        X_train (array): Mảng 2D (hoặc list của list) các đặc trưng huấn luyện.
        y_train (array): Mảng 1D (hoặc list) các nhãn huấn luyện.
        """
        # Chuyển đổi sang mảng NumPy để dễ dàng tính toán
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        # print(f"Mô hình đã 'fit' (lưu trữ) {len(self.X_train)} điểm dữ liệu.")

    def predict(self, X_test):
        """
        Dự đoán nhãn cho một tập dữ liệu kiểm thử (có thể có nhiều mẫu).
        
        Parameters:
        X_test (array-like): Mảng 2D (hoặc list của list) các đặc trưng cần dự đoán.
        
        Returns:
        list: Danh sách các nhãn dự đoán cho từng mẫu trong X_test.
        """
        X_test = np.array(X_test)
        predictions = []
        
        # Lặp qua từng mẫu trong tập test để dự đoán
        for test_sample in X_test:
            prediction = self._predict_one(test_sample)
            predictions.append(prediction)
        
        return predictions

    def _predict_one(self, test_sample):
        """
        Hàm nội bộ: Dự đoán cho một điểm dữ liệu (test_sample) đơn lẻ.
        Đây là logic cốt lõi từ file gốc của bạn, nhưng được tối ưu hóa.
        """
        
        # 1. Tính toán khoảng cách (Euclidean)
        # Sử dụng NumPy broadcasting để tính toán hiệu quả, nhanh hơn vòng lặp
        distances = np.sqrt(np.sum((self.X_train - test_sample)**2, axis=1))
        
        # 2. Tìm K chỉ số (indices) của hàng xóm gần nhất
        # np.argsort() trả về CHỈ SỐ của các phần tử đã sắp xếp
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # 3. Lấy nhãn của K hàng xóm đó
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        
        # 4. Bỏ phiếu (Voting) để quyết định nhãn cuối cùng
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, X_test, y_test):
        """
        Phương thức "thủ công" để tính toán accuracy, 
        thay thế cho hàm test() ở file gốc.
        """
        # 1. Lấy dự đoán từ hàm predict của chính class này
        y_pred = self.predict(X_test)
        
        # 2. Chuyển đổi sang NumPy để dễ so sánh (nếu chưa)
        y_test = np.array(y_test)
        
        # 3. Tính toán accuracy (đúng như logic hàm test cũ)
        correct = 0
        total = len(y_test)
        
        for i in range(total):
            if y_pred[i] == y_test[i]:
                correct += 1
                
        # 4. Trả về kết quả
        accuracy = correct / total
        return accuracy
    def get_params(self, deep=True):
        """
        GridSearchCV sẽ gọi hàm này để xem mô hình có tham số gì.
        Nó phải trả về một dict chứa tên tham số (giống hệt __init__)
        và giá trị hiện tại của nó.
        """
        return {'k': self.k}

    #  HÀM set_params 

    def set_params(self, **params):
        """
        GridSearchCV sẽ gọi hàm này để THAY ĐỔI tham số 'k'.
        Ví dụ: tuner.set_params(k=7)
        """
        if 'k' in params:
            # Cập nhật giá trị 'k' của class
            self.k = params['k']
            
        # Hàm này cũng BẮT BUỘC phải 'return self'
        return self
    # --- CÁC HÀM CŨ BỊ LOẠI BỎ ---

    # def __init__(self, data_set, k):
    #   -> ĐÃ BỊ THAY THẾ. Dữ liệu không còn được đưa vào lúc khởi tạo.
