from LinearRegressionSkeleton import LinearRegression

model = LinearRegression()

model.fit(model.X_train, model.y_train)
model.predict(model.X_test)