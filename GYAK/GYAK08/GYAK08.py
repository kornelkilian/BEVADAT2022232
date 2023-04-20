from LinearRegressionSkeleton import LinearRegression

model = LinearRegression()

model.fit(model.X, model.y)
model.predict(model.X_test)