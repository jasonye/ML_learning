Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.

参考文档：https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

* train_test_split：用户拆分数据为训练数据和验证数据

### 安装和使用

选择对应的模型

```python
pip3 install scikit-learn

from sklearn.tree import DecisionTreeRegressor


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
#X.describe()
X.head()
y = melbourne_data.Price
#y.describe()


# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```
