import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits as digits


class KMeansOnDigits:
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def load_dataset(self):
        self.digits = digits()
        
    def predict(self):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = self.model.fit_predict(self.digits.data)
        
    def get_labels(self):
        self.labels = np.zeros_like(self.clusters)
        for i in range(self.n_clusters):
            mask = (self.clusters == i)
            self.labels[mask] = mode(self.digits.target[mask])[0]
        
    def calc_accuracy(target_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
        return round(accuracy_score(target_labels, predicted_labels), 2)
        
    def confusion_matrix(self):
        self.mat = confusion_matrix(self.labels, self.clusters)
