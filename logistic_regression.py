from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# load CSV
print('load the data')
data = pd.read_csv("week_shipments_quantity.csv", header=None, names=["week", "year", "companyName", "warehouseID", "count"], dtype={"companyName": str})
# transform strings into values to be compatible with strings
data = pd.get_dummies(data, columns=["companyName", "warehouseID"], prefix=["companyName", "warehouseID"])
print(f'got {data.count()} rows to learn')



print('get feature and target')
# Prepare data features (inputs) and output(target)
features = data[["week", "year"] + list(data.columns[5:])].values
target = data["count"]


# Split the data into training and validation sets (I set to 20% for validation)
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=83)

print(f'Splitted the data {len(X_val)} to learn and {len(y_val)} to test')


# Choose logistic regression as the model (step 5)
model = LogisticRegressionCV()

print('running model fit')
# Train the model (step 6)
model.fit(X_train, y_train)
print('finish fit')


# Evaluate the model (step 7)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
classification = classification_report(y_val, y_pred)
print(f"Accuracy: {accuracy}")
print(classification)





# # Now, you can use the model to make predictions for a future date (step 8)
# future_date = ...
# future_warehouse = ...
# future_user = ...
# future_gift_prediction = model.predict([[future_date, future_warehouse, future_user]])