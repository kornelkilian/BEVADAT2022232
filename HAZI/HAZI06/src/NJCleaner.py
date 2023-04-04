import pandas as pd

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
        bins = pd.IntervalIndex.from_tuples([(0, 3.59), (4, 7.59), (8, 11.59), (12, 15.59), (16, 19.59), (20, 23.59)])
        labels = ['late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night']
        self.data['part_of_the_day'] = pd.cut(pd.to_datetime(self.data['scheduled_time']).dt.hour, bins=bins, labels=labels)
        self.data.drop(['scheduled_time'], axis=1, inplace=True)
        return self.data
    
    def convert_delay(self):
        self.data['delay'] = self.data['delay_minutes'].apply(lambda x: 0 if x < 5 else 1)
        return self.data
    
    def drop_unnecessary_columns(self):
        self.data.drop(columns=['train_id', 'actual_time', 'delay_minutes'], axis=1, inplace=True)
        return self.data
    
    def save_first_60k(self, path):
        self.data.iloc[:60000].to_csv(path, index=False)
    
    def prep_df(self):
        self.order_by_scheduled_time()
        self.drop_columns_and_nan()
        self.convert_date_to_day()
        self.convert_scheduled_time_to_part_of_the_day()
        self.convert_delay()
        self.drop_unnecessary_columns()
        return self.data

