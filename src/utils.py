# src/utils.py

import pandas as pd

def categorize_weather(weather_code):
    """
    دالة لتصنيف الطقس بناءً على التصنيفات الرقمية.
    
    Args:
        weather_code (int): رمز الطقس الرقمي.
    
    Returns:
        str: تصنيف الطقس.
    """
    weather_mapping = {
        0: 'Sunny',
        1: 'Clear',
        2: 'Scattered clouds',
        3: 'Passing clouds',
        4: 'Partly sunny',
        5: 'Low level haze',
        6: 'Fog',
        7: 'Rain Passing clouds',
        8: 'Thunderstorms Passing clouds',
        9: 'Overcast',
        10: 'Mild',
        11: 'Duststorm',
        12: 'Light rain Overcast',
        13: 'Rain Overcast',
        14: 'Rain Partly sunny',
        15: 'Light rain Partly sunny',
        16: 'Broken clouds'
    }
    return weather_mapping.get(weather_code, 'Other')  # تعيين 'Other' إذا لم يتم العثور على التصنيف
