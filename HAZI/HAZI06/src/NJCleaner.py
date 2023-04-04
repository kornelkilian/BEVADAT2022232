import pandas as pd
from datetime import datetime


class NJCleaner:
    
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
    
    def order_by_scheduled_time(self):
        self.data.sort_values(by='scheduled_time', inplace=True)
        return self.data
    
    def drop_columns_and_nan(self):
        self.data.drop(['from', 'to'], axis=1, inplace=True)
        self.data.dropna(inplace=True)
        return self.data
    
    def convert_date_to_day(self):
        self.data['day'] = pd.to_datetime(self.data['date']).dt.day_name()
        self.data.drop(columns='date', axis=1, inplace=True)
        return self.data
    

    def convert_scheduled_time_to_part_of_the_day(self):
        def get_part_of_day(hour):
            if 4 <= hour < 8:
                return 'early_morning'
            elif 8 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 16:
                return 'afternoon'
            elif 16 <= hour < 20:
                return 'evening'
            elif 20 <= hour <= 23:
                return 'night'
            else:
                return 'late_night'

        self.data['scheduled_time'] = self.data['scheduled_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').time())
        self.data['part_of_the_day'] = self.data['scheduled_time'].apply(lambda x: get_part_of_day(x.hour))
        self.data.drop('scheduled_time', axis=1, inplace=True)
        return self.data
    
    def convert_delay(self):
        self.data['delay'] = self.data['delay_minutes'].apply(lambda x: 0 if x < 5 else 1)
        return self.data
    
    def drop_unnecessary_columns(self):
        self.data.drop(columns=['train_id', 'actual_time', 'delay_minutes'], axis=1, inplace=True)
        return self.data
    
    def save_first_60k(self, path='data/NJ.csv'):
        self.data.iloc[:60000].to_csv(path, index=False)
    
    def prep_df(self):
        self.order_by_scheduled_time()
        self.drop_columns_and_nan()
        self.convert_date_to_day()
        self.convert_scheduled_time_to_part_of_the_day()
        self.convert_delay()
        self.drop_unnecessary_columns()
        self.save_first_60k()
        return self.data

