from sklearn import linear_model

LR = linear_model.LinearRegression()

x = [[1, 3], [4, 2], [5, 1], [7, 4], [8, 9]]
y = [1.002, 4.1, 4.96, 6.78, 8.2]

LR.fit(x, y)

k = LR.coef_
b = LR.intercept_

print(k, b)

print(LR.predict([[1, 2]]))