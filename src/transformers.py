# src/transformers.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import categorize_weather  # تأكد من أن المسار صحيح بناءً على هيكلية المشروع

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    محول مخصص لهندسة الميزات.
    """
    def __init__(self, holidays_file='data/holidays.csv'):
        """
        تهيئة المحول مع ملف العطلات.
        
        Args:
            holidays_file (str): مسار ملف CSV يحتوي على تواريخ العطلات.
        """
        self.holidays_file = holidays_file
        self.holidays = self.load_holidays()

    def load_holidays(self):
        """
        تحميل تواريخ العطلات من ملف CSV.
        
        Returns:
            list: قائمة من تواريخ العطلات كـ pd.Timestamp.
        """
        try:
            holidays_df = pd.read_csv(self.holidays_file)
            if 'StartDate' in holidays_df.columns and 'EndDate' in holidays_df.columns:
                # تحويل StartDate و EndDate إلى datetime
                holidays_df['StartDate'] = pd.to_datetime(holidays_df['StartDate'])
                holidays_df['EndDate'] = pd.to_datetime(holidays_df['EndDate'])
                # إنشاء قائمة بكل تواريخ العطلات بين StartDate و EndDate
                all_holidays = []
                for _, row in holidays_df.iterrows():
                    start = row['StartDate']
                    end = row['EndDate']
                    all_holidays.extend(pd.date_range(start=start, end=end).tolist())
                return [date.date() for date in all_holidays]  # تحويل إلى date فقط
            elif 'Date' in holidays_df.columns:
                return pd.to_datetime(holidays_df['Date']).dt.date.tolist()
            else:
                raise ValueError("يجب أن يحتوي ملف العطلات على عمود 'Date' أو كلا من 'StartDate' و 'EndDate'.")
        except Exception as e:
            print(f"حدث خطأ أثناء تحميل ملف العطلات: {e}")
            return []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # تحويل 'Timestamp' إلى datetime
        X['Timestamp'] = pd.to_datetime(X['Timestamp'], errors='coerce')

        # هندسة الميزات الأساسية
        X['DayOfWeek'] = X['Timestamp'].dt.weekday
        X['Hour'] = X['Timestamp'].dt.hour
        X['IsWeekend'] = X['DayOfWeek'].isin([4, 5]).astype(int)  # 4=Friday, 5=Saturday
        X['Date'] = X['Timestamp'].dt.date
        X['IsHoliday'] = X['Date'].apply(lambda x: int(x in self.holidays))
        X['WeatherCat'] = X['Weather'].apply(categorize_weather)
        X['Temp_Humidity_Interaction'] = X['Temp'] * X['Humidity']
        X['WeekOfYear'] = X['Timestamp'].dt.isocalendar().week.astype(int)
        X['DayOfYear'] = X['Timestamp'].dt.dayofyear
        X['TimeSinceLastEvent'] = X.groupby('PersonID')['Timestamp'].diff().dt.total_seconds() / 3600
        X['TimeSinceLastEvent'] = X['TimeSinceLastEvent'].fillna(0)
        X['LogTimeSinceLastEvent'] = np.log1p(X['TimeSinceLastEvent'])

        # ترميز One-Hot للميزات الفئوية
        X = pd.get_dummies(X, columns=['WeatherCat'], drop_first=False)

        # إضافة ميزات إضافية بالقيم الافتراضية
        X['AvgDurationPerPerson'] = 0  # سيتم تعيينها إلى 0 لأنها تتطلب بيانات تاريخية
        X['SameDayDurationDiffs'] = 0
        X['AvgDurationPerDayOfWeek'] = 0
        X['AvgDurationPerHour'] = 0
        for lag in [1, 2, 3]:
            X[f'Duration_Hours_Lag{lag}'] = 0
        for window in [3, 5]:
            X[f'Duration_Hours_RollingMean{window}'] = 0

        return X
