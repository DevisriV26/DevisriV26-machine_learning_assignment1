import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error


# ----------------------------
# 1 Load Dataset
# ----------------------------

data = pd.read_csv("laptop.csv")

print("First 5 Rows:")
print(data.head())

print("\nColumns in dataset:")
print(data.columns)


# ----------------------------
# 2 Remove unnecessary columns
# ----------------------------

data = data.drop(columns=['Unnamed: 0','Model','Graphics','OS','Warranty'])


# ----------------------------
# 3 Clean Price column
# ----------------------------

data['Price'] = data['Price'].replace('[₹,]', '', regex=True)
data['Price'] = data['Price'].astype(float)


# ----------------------------
# 4 Convert RAM to numeric
# ----------------------------

data['Ram'] = data['Ram'].str.extract(r'(\d+)').astype(float)


# ----------------------------
# 5 Convert SSD to numeric
# ----------------------------

data['SSD'] = data['SSD'].str.extract(r'(\d+)').astype(float)


# ----------------------------
# 6 Convert Generation
# ----------------------------

data['Generation'] = data['Generation'].str.extract(r'(\d+)').astype(float)


# ----------------------------
# 7 Convert Core
# ----------------------------

data['Core'] = data['Core'].str.extract(r'(\d+)').astype(float)


# ----------------------------
# 8 Convert Display size
# ----------------------------

data['Display'] = data['Display'].str.extract(r'(\d+\.?\d*)').astype(float)


# ----------------------------
# 9 Convert Rating
# ----------------------------

data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')


# ----------------------------
# 10 Handle missing values
# ----------------------------

data = data.fillna(data.mean(numeric_only=True))


# ----------------------------
# 11 Define X and Y
# ----------------------------

X = data.drop('Price', axis=1)
y = data['Price']


# ----------------------------
# 12 Split dataset
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# 13 Linear Regression
# ----------------------------

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)

print("\nLinear Regression MSE:", lr_mse)


# ----------------------------
# Linear Regression Graph
# ----------------------------

plt.figure()

plt.scatter(y_test, lr_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Linear Regression Prediction")

plt.show()


# ----------------------------
# 14 Decision Tree
# ----------------------------

dt = DecisionTreeRegressor(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_pred)

print("Decision Tree MSE:", dt_mse)


# ----------------------------
# Decision Tree Graph
# ----------------------------

plt.figure(figsize=(18,8))

plot_tree(
    dt,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Visualization")

plt.show()


# ----------------------------
# 15 Prediction Table
# ----------------------------

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Linear Regression": lr_pred,
    "Decision Tree": dt_pred
})

print("\nSample Predictions:")
print(results.head())
