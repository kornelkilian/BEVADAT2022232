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
        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
        self.X = self.df['petal width (cm)'].values
        self.y = self.df['sepal length (cm)'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        

    def fit(self, X: np.array, y: np.array):

        # Building the model
        self.m = 0
        self.c = 0

        L = self.lr  # The learning Rate
        epochs = self.epoch  # The number of iterations to perform gradient descent

        n = float(len(self.X_train)) # Number of elements in X

        # Performing Gradient Descent 
        losses = []
        for i in range(epochs): 
            y_pred = self.m*self.X_train + self.c  # The current predicted value of Y

            residuals = y_pred - self.y_train
            loss = np.sum(residuals ** 2)
            losses.append(loss)
            D_m = (-2/n) * sum(self.X_train * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m + L * D_m  # Update m
            self.c = self.c +L * D_c  # Update c
            if i % 100 == 0:
                print(np.mean(self.y_train-y_pred))
        

    def predict(self, X):
        y_pred = self.m*self.X_test + self.c

        return y_pred
