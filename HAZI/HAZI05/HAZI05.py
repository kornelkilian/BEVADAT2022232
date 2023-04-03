import pandas as pd
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import numpy as np

class KNNClassifier:
    def __init__(self, k: int, test_split_ratio: float):
        self.k = k
        self.test_split_ratio = test_split_ratio

    @staticmethod
    def load_csv(csv_path:str) ->Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = pd.read_csv(csv_path, delimiter=',')
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x,y = pd.DataFrame(dataset.iloc[:,:8]), pd.DataFrame(dataset.iloc[:,-1])
        return x, y
    
    def train_test_split(self, features: pd.DataFrame, labels: pd.DataFrame) -> None:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"
        self.x_train, self.y_train = features[:train_size], labels[:train_size]
        self.x_test, self.y_test = features[train_size:train_size + test_size], labels[train_size:train_size + test_size]

    def euclidean(self, element_of_x:pd.DataFrame) -> pd.DataFrame:
        points = self.x_train.reset_index(drop=True)
        element_of_x = element_of_x.reindex(points.index).ffill()
        distances = ((points - element_of_x) ** 2).sum(axis=1) ** 0.5
        return pd.DataFrame(distances, columns=['distance'])
    
    def predict(self, x_test: pd.DataFrame) -> None:
        labels_pred = []
        for i, x_test_element in x_test.iterrows():
            distances = self.euclidean(x_test_element)
            distances = pd.concat([distances, self.y_train], axis=1)
            distances = distances.sort_values(by=0, axis=0).iloc[:self.k, 1]
            label_pred = mode(distances, axis=None).mode[0]
            labels_pred.append(label_pred)
        self.y_preds = pd.Series(labels_pred, dtype=pd.np.int32)

    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def best_k(self) -> float:
        old_k = self.k
        results = []
        for k in range(1,21):
            self.k = k
            self.predict(self.x_test)
            res = round(self.accuracy(), 2)
            results.append((k, res))
        self.k = old_k
        return max(results, key=lambda x:x[1])

    def confusion_matrix(self) -> pd.DataFrame:
        return confusion_matrix(self.y_test, self.y_preds)

    @property
    def k_neighbors(self) -> int:
        return self.k
