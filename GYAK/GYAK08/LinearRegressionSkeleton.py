import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epoch=epochs
        self.lr=lr
        self.m=0
        self.c=0
        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
        self.X = self.df['petal width (cm)'].values
        self.y = self.df['sepal length (cm)'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        

    def fit(self, X: np.array, y: np.array):
            n = float(len(X))
            self.losses = []
            for i in range(self.epochs): 
                y_pred = self.m*X + self.c 
                residuals = y_pred - y
                loss = np.sum(residuals ** 2)
                self.losses.append(loss)
                D_m = (-2/n) * sum(X * residuals)  
                D_c = (-2/n) * sum(residuals)  
                self.m = self.m + self.lr * D_m  
                self.c = self.c + self.lr * D_c  

        

    def predict(self, X):
        y_pred = self.m*self.X_test + self.c

        return y_pred
