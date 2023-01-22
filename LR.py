import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

datas = datasets.load_boston()
# print(datas.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
# print(datas.DESCR)
# # details about data attribute and data
datas_x = datas.data[:,np.newaxis,2]
# # (there 2 is index)(: this mean all)(np.newaxis will from an array )
# print(datas_x)
datas_x_train = datas_x[:-50]
datas_x_test =datas_x[-30:]
datas_y_test =datas.target[-30:]
datas_y_train = datas.target[:-50]
model = linear_model.LinearRegression()
model.fit(datas_x_train,datas_y_train)
pridict_y = model.predict(datas_x_test)
# pridict values of  y on bases of x
print("mean squared error :",mean_squared_error(datas_y_test,pridict_y))
# avg error percentage of pridiction vs real value
print("weight:",model.coef_)
print("intercept:",model.intercept_)
# intercept of  with axis
plt.scatter(datas_x_test,datas_y_test)
# to plot data point
plt.plot(datas_x_test,pridict_y)
# line of pridiction with decresing mse
plt.show()