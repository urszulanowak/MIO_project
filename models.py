import keras
from keras import layers
from keras import models
import numpy as np


def build_encoder(input_shape, encoding_layers):
    input_layer = layers.Input(shape=input_shape)
    x = input_layer
    for units, activation in encoding_layers:
        x = layers.Dense(units, activation=activation)(x)
    model = models.Model(input_layer, x, name="encoder")
    return model


def build_decoder(encoding_dim, decoding_layers):
    encoded_input = layers.Input(shape=encoding_dim)
    x = encoded_input
    for units, activation in decoding_layers:
        x = layers.Dense(units, activation=activation)(x)
    model = models.Model(encoded_input, x, name="decoder")
    return model


def build_autoencoder(input_dim, encoder, decoder):
    autoencoder_input = layers.Input(shape=(input_dim,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = models.Model(inputs=autoencoder_input, outputs=decoded, name="autoencoder")
    return autoencoder