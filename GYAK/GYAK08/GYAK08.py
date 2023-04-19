from LinearRegressionSkeleton import LinearRegression

model = LinearRegression(3000, 0.003)

model.fit(model.X, model.y)
model.predict(model.X_test)