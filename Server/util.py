import json
import pickle
import numpy as np

# global variables --> 
locations = None
data_columns = None
model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)


def get_location_names():
    return locations


# just loading the artifacts/data from the json and pickle file -->
def load_artifacts():
    global data_columns
    global locations

    print("All artifacts loading... ")
    with open("./artifacts/columns.json", 'r') as f:
        data_columns = json.load(f)['data_columns']  # whatever objects are loaded will be converted to dictionary
        locations = data_columns[3:]

    global model
    with open("./artifacts/house_price_prediction_model.pickle", 'rb') as f:
        model = pickle.load(f)

    print("All artifacts loaded successfully...")


if __name__ == '__main__':
    load_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
    print(get_estimated_price('Ejipura', 1000, 2, 2))


