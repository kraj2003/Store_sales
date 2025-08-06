import pandas as pd
import numpy as np

def load_data():
    """Load the training data"""
    df = pd.read_csv('../data/train.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def create_daily_sales(df):
    """Convert to daily sales totals"""
    daily = df.groupby('date')['sales'].sum().reset_index()
    daily.columns = ['date', 'total_sales']
    return daily.sort_values('date')

def add_features(df):
    """Add simple features for prediction"""
    df = df.copy()
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    
    # Lag features
    # yesterady sales and last week sales
    df['sales_yesterday'] = df['total_sales'].shift(1)
    df['sales_last_week'] = df['total_sales'].shift(7)
    
    # Moving averages
    # last seven days and 30 days averages
    # df['sales_avg_7d'] = df['total_sales'].rolling(7).mean()
    # df['sales_avg_30d'] = df['total_sales'].rolling(30).mean()
    
    return df

def get_features():
    """Return list of feature columns"""
    return ['year', 'month', 'day', 'dayofweek', 'is_weekend', 
            'sales_yesterday', 'sales_last_week']

def save_daily_sales(df):
    """Save processed daily sales"""
    df.to_csv('../data/daily_sales.csv', index=False)
    print("Saved daily_sales.csv")