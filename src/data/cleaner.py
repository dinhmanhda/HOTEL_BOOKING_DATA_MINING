import pandas as pd

class HotelCleaner:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        # 1. Xử lý giá trị thiếu (Tiêu chí B)
        self.df['children'] = self.df['children'].fillna(0)
        self.df['country'] = self.df['country'].fillna('Unknown')
        self.df['agent'] = self.df['agent'].fillna(0)
        self.df['company'] = self.df['company'].fillna(0)

        # 2. Xóa Data Leakage (Tiêu chí E - Quan trọng nhất)
        # Các cột này chứa thông tin về kết quả đặt phòng, phải xóa khi train model
        leakage_cols = ['reservation_status', 'reservation_status_date']
        self.df.drop(columns=leakage_cols, inplace=True, errors='ignore')
        
        return self.df