import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data_sets():
    mitbih_train = pd.read_csv('data/mitbih_train.csv', header=None)
    mitbih_test = pd.read_csv('data/mitbih_test.csv', header=None)

    x_train = mitbih_train.values[:, :-1]
    x_test = mitbih_test.values[:, :-1]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    return X_train_scaled, X_test_scaled
