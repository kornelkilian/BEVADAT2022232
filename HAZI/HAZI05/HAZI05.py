import pandas as pd
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

class KNNClassifier:
    def init(self, k: int, test_split_ratio: float):
        self.k = k
        self.test_split_ratio = test_split_ratio

    def load_csv(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = pd.read_csv(csv_path, header=None)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x, y = dataset.iloc[:, :4], dataset.iloc[:, -1]
        x = x.fillna(3.5)
        indices = (x > 13.0) | (x < 0.0)
        x = x[~indices.any(axis=1)]
        y = y[~indices.any(axis=1)]
        x_train, y_train, x_test, y_test = self.train_test_split(x, y)
        return x_train, y_train, x_test, y_test

    def train_test_split(self, features: pd.DataFrame, labels: pd.DataFrame) -> None:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"
        self.x_train, self.y_train = features[:train_size], labels[:train_size]
        self.x_test, self.y_test = features[train_size:train_size + test_size], labels[train_size:train_size + test_size]

    def euclidean(self, element_of_x: pd.DataFrame) -> pd.DataFrame:
        return pd.Series(((self.x_train - element_of_x) ** 2).sum(axis=1)).apply(pd.np.sqrt)

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

    def confusion_matrix(self) -> pd.DataFrame:
        return confusion_matrix(self.y_test, self.y_preds)

    @property
    def k_neighbors(self) -> int:
        return self.k
