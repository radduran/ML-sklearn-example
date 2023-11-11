import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

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

    def __str__(self):
        if self.interval > 60:
            return f"{self.interval/60:.2f} minutes"
        return f"{self.interval:.2f} seconds"


def get_csv_data(filename, to_learn=True):
    # Load the data
    csv_cols = ["week", "year", "companyName", "warehouseID"]
    if to_learn:
        csv_cols.append("count")
    # load CSV
    data = pd.read_csv(
        filename, header=None, names=csv_cols, dtype={"companyName": str}
    )

    # Initialize the encoders
    company_encoder = LabelEncoder()
    warehouse_encoder = LabelEncoder()

    # Fit and transform the data with the encoders
    if to_learn:
        company_encoder.fit(data["companyName"])
        warehouse_encoder.fit(data["warehouseID"])
        data["companyName"] = company_encoder.transform(data["companyName"])
        data["warehouseID"] = warehouse_encoder.transform(data["warehouseID"])
        joblib.dump(company_encoder, "company_encoder.pkl")
        joblib.dump(warehouse_encoder, "warehouse_encoder.pkl")

    # If not to_learn, it means we are preparing the data for prediction
    else:
        # Load the encoders
        company_encoder = joblib.load("company_encoder.pkl")
        warehouse_encoder = joblib.load("warehouse_encoder.pkl")
        # Transform the data with the encoders
        data["companyName"] = company_encoder.transform(data["companyName"])
        data["warehouseID"] = warehouse_encoder.transform(data["warehouseID"])

    return data


def get_features_target(data, to_learn=True):
    # Prepare data features (inputs) and output(target)
    features = data[["week", "year", "companyName", "warehouseID"]].values
    if not to_learn:
        return features, None
    return features, data["count"]


def get_train_test(features, target):
    # Split the data into training and validation sets (I set to 20% for validation)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=83
    )
    return X_train, X_test, y_train, y_test


def get_model(X_train, y_train, estimators=2, override=False):
    # if model already exists, load it
    if not override:
        try:
            model= joblib.load(f"model.pkl_{estimators}")
            return model
        except:
            print("model not found, creating a new one")

    # Initialize the Random Forest regressor
    rf = RandomForestRegressor(n_estimators=estimators, random_state=42)
    print("running model fit")
    # Train the model
    with Timer():
        rf.fit(X_train, y_train)
    print("finish fit")

    # save model
    joblib.dump(rf, f"model.pkl_{estimators}")
    return rf


def prepare_model(iterations, get_score=False, show_statistics=True ):
    # get data from CSV
    data = get_csv_data("week_shipments_quantity.csv")

    # split data into features and target
    features, target = get_features_target(data)

    # get train and test data from features and target
    X_train, X_test, y_train, y_test = get_train_test(features, target)

    # do the magic measuring the time, later we can use this to compare with other estimators
    with Timer() as timer:
        rf = get_model(X_train, y_train, estimators=iterations)

    if get_score:
        print(f"fit time: {timer}")
        # Evaluate the model
        predictions = rf.predict(X_test)
                    
        if show_statistics:
            # Calculate the mean squared error
            print("Statics for:", iterations)
            mse = mean_squared_error(y_test, predictions)
            print("Mean Squared Error:", mse)
            print("Root Mean Squared Error:", mse ** (1 / 2))
            print("Score:", rf.score(X_test, y_test))

        return rf, rf.score(X_test, y_test)

        

    return rf



def predict(week, year, warehouseID, companyName):
    # Convert the parameters to a DataFrame
    data = pd.DataFrame(
        {
            "week": [week],
            "year": [year],
            "warehouseID": [warehouseID],
            "companyName": [companyName],
        }
    )
    company_encoder = joblib.load("company_encoder.pkl")
    warehouse_encoder = joblib.load("warehouse_encoder.pkl")

    # Encode 'warehouseID' and 'companyName'
    data["warehouseID"] = warehouse_encoder.transform([warehouseID])
    data["companyName"] = company_encoder.transform([companyName])

    # Extract features from the data
    features, _ = get_features_target(data, to_learn=False)

    # Make a prediction
    prediction = rf.predict(features)

    print(f"Predicted for {companyName} warehouse {warehouseID} on the week {week} of {year} count: {prediction[0]}")
    
    return prediction[0]


tries = [200, 225, 250, 275 ,300]


scores = {}
for t in tries:
    print(f"--------------------------------{t}---------------------------------------")
    rf, score = prepare_model(t, get_score=True, show_statistics=True)
    scores[t] = score
    
print(scores)

# Make a prediction
# LLC_prediction = predict(
#     week=45,
#     year=2024,
#     warehouseID="8af560db-84eb-4504-bd94-6a933c44945e",
#     companyName="EMP Strategies LLC",
# )

