from regression_model import LR, LRwN

x = [[1, 3], [4, 2], [5, 1], [7, 4], [8, 9]]
y = [1.002, 4.1, 4.96, 6.78, 8.2]

reg = LR(x=x, y=y, i=100000, lr=0.01)
regn = LRwN(x=x, y=y, i=100000, lr=0.01, alpha=0.001)

reg.fit(True)
regn.fit(True)

print(reg.predict([[1, 2]]))
print(regn.predict([[1, 2]]))