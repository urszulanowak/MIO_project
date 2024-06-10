import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data_sets():
    mitbih_train = pd.read_csv(os.path.join('data', 'mitbih_train.csv'), header=None)
    mitbih_test = pd.read_csv(os.path.join('data', 'mitbih_test.csv'), header=None)

    print("mitbih_train shape:", mitbih_train.shape)  # Debugging line
    print("mitbih_test shape:", mitbih_test.shape)    # Debugging line

    x_train = mitbih_train.values[:, :-1]
    x_test = mitbih_test.values[:, :-1]

    # Check if x_train and x_test have features
    if x_train.shape[1] == 0 or x_test.shape[1] == 0:
        raise ValueError("Input data must have at least one feature.")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    return X_train_scaled, X_test_scaled
