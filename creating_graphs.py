from data import load_data_sets
from model_testing import closest, model_testing
from keras.optimizers import Adam
from models import build_autoencoder, build_decoder, build_encoder
from main import main
import matplotlib.pyplot as plt
import numpy as np
import keras
import statistics
from pathlib import Path


train_set, test_set = load_data_sets()

test_set1=test_set[:2000]
train_set=train_set[:2000]



main("base",X_train=train_set,X_test=test_set1)
model_testing("base", test_set)

main("ep200",X_train=train_set,X_test=test_set1,epoch_count=200)
model_testing("ep200", test_set)


main("encode16",X_train=train_set,X_test=test_set1,epoch_count=100)
model_testing("encode16", test_set)

main("encode8",X_train=train_set,X_test=test_set1,epoch_count=100)
model_testing("encode8", test_set)

main("encode64",X_train=train_set,X_test=test_set1,epoch_count=100)
model_testing("encode64", test_set)

# layers1 = [(16, 'relu'), (32, 'relu'), (32, 'relu')]
# layers2 = [(32, 'relu'), (8, 'relu'), (32, 'relu')]
# layers3 = [(8, 'relu'), (8, 'relu'), (32, 'relu')]
# layers4 = [(32, 'relu'), (16, 'relu'), (32, 'relu')]

# main("lay1",X_train=train_set,X_test=test_set1,encoding_layers=layers1)
# model_testing("lay1", test_set)

# main("lay2",X_train=train_set,X_test=test_set1,encoding_layers=layers2)
# model_testing("lay2", test_set)

# main("lay3",X_train=train_set,X_test=test_set1,encoding_layers=layers3)
# model_testing("lay3", test_set)

# main("lay4",X_train=train_set,X_test=test_set1,encoding_layers=layers4)
# model_testing("lay4", test_set)

