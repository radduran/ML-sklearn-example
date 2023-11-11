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
        print(f"Elapsed time: {self.interval/60:.2f} minutes")



def prepare_new_data(category_map, week, year, companyName, warehouseID):
    # Initialize a new DataFrame with the same columns as the training data
    new_data = pd.DataFrame(columns=data.columns)

    # Set the week and year
    new_data['week'] = [week]
    new_data['year'] = [year]

    # Set the one-hot encoded columns
    for column, mapping in category_map.items():
        # Get the encoded value for the company name and warehouse ID
        encoded_companyName = get_original_value(mapping, 'companyName', companyName)
        encoded_warehouseID = get_original_value(mapping, 'warehouseID', warehouseID)

        # Set the corresponding columns in the new data
        new_data[f'{column}_{encoded_companyName}'] = 1
        new_data[f'{column}_{encoded_warehouseID}'] = 1

    return new_data

def prepare_category_map(data):
    # Extract the columns that were one-hot encoded
    encoded_columns = [col for col in data.columns if col.startswith("companyName_") or col.startswith("warehouseID_")]

    # Create a mapping for each encoded column
    mapping = {}
    for col in encoded_columns:
        original_col = col.split('_')[0]  # Get the original column name
        unique_encoded_values = data[col].unique()  # Get unique encoded values
        mapping[original_col] = {value: data[data[col] == value][original_col].iloc[0] for value in unique_encoded_values}
    return mapping


def get_original_value(map, column, encoded_value):
    return map[column][encoded_value]

def get_data(filename, to_learn=True):
    # Load the data
    csv_cols=["week", "year", "companyName", "warehouseID"]
    if to_learn:
        csv_cols.append("count")    
    # load CSV
    print('load the data')
    data = pd.read_csv(filename, header=None, names=csv_cols, dtype={"companyName": str})


    # Initialize the encoders
    company_encoder = LabelEncoder()
    warehouse_encoder = LabelEncoder()

    # Fit and transform the data with the encoders
    if to_learn:
        company_encoder.fit(data['companyName'])
        warehouse_encoder.fit(data['warehouseID'])
        data['companyName'] = company_encoder.transform(data['companyName'])
        data['warehouseID'] = warehouse_encoder.transform(data['warehouseID'])
        
        joblib.dump(company_encoder, 'company_encoder.pkl')
        joblib.dump(warehouse_encoder, 'warehouse_encoder.pkl')


    # If not to_learn, it means we are preparing the data for prediction
    else:
        # Load the encoders
        company_encoder = joblib.load('company_encoder.pkl')
        warehouse_encoder = joblib.load('warehouse_encoder.pkl')

        # Transform the data with the encoders
        data['companyName'] = company_encoder.transform(data['companyName'])
        data['warehouseID'] = warehouse_encoder.transform(data['warehouseID'])

    return data, company_encoder, warehouse_encoder

def get_features_target(data, to_learn=True):
    print('get feature and target')
    # Prepare data features (inputs) and output(target)
def get_data(filename, to_learn=True):
    # Load the data
    csv_cols=["week", "year", "companyName", "warehouseID"]
    if to_learn:
        csv_cols.append("count")    
    # load CSV
    print('load the data')
    data = pd.read_csv(filename, header=None, names=csv_cols, dtype={"companyName": str})


    # Initialize the encoders
    company_encoder = LabelEncoder()
    warehouse_encoder = LabelEncoder()

    # Fit and transform the data with the encoders
    if to_learn:
        company_encoder.fit(data['companyName'])
        warehouse_encoder.fit(data['warehouseID'])
        data['companyName'] = company_encoder.transform(data['companyName'])
        data['warehouseID'] = warehouse_encoder.transform(data['warehouseID'])
        
        joblib.dump(company_encoder, 'company_encoder.pkl')
        joblib.dump(warehouse_encoder, 'warehouse_encoder.pkl')


    # If not to_learn, it means we are preparing the data for prediction
    else:
        # Load the encoders
        company_encoder = joblib.load('company_encoder.pkl')
        warehouse_encoder = joblib.load('warehouse_encoder.pkl')

        # Transform the data with the encoders
        data['companyName'] = company_encoder.transform(data['companyName'])
        data['warehouseID'] = warehouse_encoder.transform(data['warehouseID'])

    return data, company_encoder, warehouse_encoder

def get_features_target(data, to_learn=True):
    print('get feature and target')
    # Prepare data features (inputs) and output(target)
    features = data[["week", "year"] + list(data.columns[5:])].values
    target = None
    if to_learn:
        target = data["count"]
        
    return features, target
def get_data(filename, to_learn=True):
    # Load the data
    csv_cols=["week", "year", "companyName", "warehouseID"]
    if to_learn:
        csv_cols.append("count")    
    # load CSV
    print('load the data')
    data = pd.read_csv(filename, header=None, names=csv_cols, dtype={"companyName": str})


    # Initialize the encoders
    company_encoder = LabelEncoder()
    warehouse_encoder = LabelEncoder()

    # Fit and transform the data with the encoders
    if to_learn:
        company_encoder.fit(data['companyName'])
        warehouse_encoder.fit(data['warehouseID'])
        
        joblib.dump(company_encoder, 'company_encoder.pkl')
        joblib.dump(warehouse_encoder, 'warehouse_encoder.pkl')
    # If not to_learn, it means we are preparing the data for prediction
    else:
        # Load the encoders
        company_encoder = joblib.load('company_encoder.pkl')
        warehouse_encoder = joblib.load('warehouse_encoder.pkl')

    # Transform the data with the encoders
    data['companyName'] = company_encoder.transform(data['companyName'])
    data['warehouseID'] = warehouse_encoder.transform(data['warehouseID'])
    return data

def get_features_target(data, to_learn=True):
    print('get feature and target')
    # Prepare data features (inputs) and output(target)
    features = data[["week", "year"] + list(data.columns[5:])].values
    target = None
    if to_learn:
        target = data["count"]
        
    return features, target

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
    rf = get_model(X_train, y_train, estimators=500, override=True)

# Evaluate the model
predictions = rf.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", mse**(1/2))
print("Score:", rf.score(X_test, y_test))

print('--------------------------------------------------------------------------------')
# now lets see hackathon week but in 2024
# Set the new data to predict
new_data_from_csv = get_data('to_predict.csv', to_learn=False)
new_data_to_predict, _ = get_features_target(new_data_from_csv, to_learn=False)
new_prediction = rf.predict(new_data_to_predict)
print(f'prediction for 2024 week 1: {new_prediction}')