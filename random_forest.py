import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import time



# timer to measure fit time
class Timer:
    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.end = time.monotonic()
        self.interval = self.end - self.start
        print(f"Elapsed time: {self.interval/60:.2f} minutes")


def get_data(csv_file, to_learn=True):   
    csv_cols=["week", "year", "companyName", "warehouseID"]
    if to_learn:
        csv_cols.append("count")
    
    # load CSV
    print('load the data')
    data = pd.read_csv(csv_file, header=None, names=csv_cols, dtype={"companyName": str})
    # transform strings into values to be compatible with strings
    data = pd.get_dummies(data, columns=["companyName", "warehouseID"], prefix=["companyName", "warehouseID"])
    print(f'got {data.count()} rows to learn')
    return data

def get_features_target(data, to_learn=True):
    print('get feature and target')
    # Prepare data features (inputs) and output(target)
    features = data[["week", "year"] + list(data.columns[5:])].values
    target = None
    if to_learn:
        target = data["count"]
        
    return features, target

def get_train_test(features, target):
    # Split the data into training and validation sets (I set to 20% for validation)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=83)
    return X_train, X_test, y_train, y_test

def get_model(X_train, y_train, estimators=2, override=False):
    # if model already exists, load it
    if not override:
        try:
            return joblib.load(f'model.pkl_{estimators}')
        except:
            print('model not found, creating a new one')
    
    # Initialize the Random Forest regressor
    rf = RandomForestRegressor(n_estimators=estimators, random_state=42)
    print('running model fit')
    # Train the model
    with Timer():
        rf.fit(X_train, y_train)
    print('finish fit')
    
    # save model
    joblib.dump(rf, f'model.pkl_{estimators}')
    return rf

# get data from CSV
data = get_data('week_shipments_quantity.csv')
# split data into features and target
features, target = get_features_target(data)

# get train and test data from features and target
X_train, X_test, y_train, y_test = get_train_test(features, target)

# do the magic measuring the time, later we can use this to compare with other estimators
with Timer():
    rf = get_model(X_train, y_train, estimators=2)

# Evaluate the model
predictions = rf.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", mse**(1/2))
print("Score:", rf.score(X_test, y_test))

print('--------------------------------------------------------------------------------')
# now lets see hackathon week but in 2024
new_data_to_predict = get_data('to_predict.csv', to_learn=False)
features_to_predict, _ = get_features_target(new_data_to_predict, to_learn=False)
prediction = rf.predict(features_to_predict)

