import keras
from keras import layers
from keras import models
import numpy as np

def build_encoder(input_dim, encoding_dim):
    input_layer = layers.Input(shape=(input_dim,))

    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
    encoder = models.Model(input_layer, encoded, name="encoder")
    return encoder


def build_decoder(encoding_dim, output_dim):
    encoded_input = layers.Input(shape=(encoding_dim,))

    decoded = layers.Dense(64, activation='relu')(encoded_input)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(output_dim, activation='sigmoid')(decoded)
    
    decoder = models.Model(encoded_input, decoded, name="decoder")
    return decoder


def build_autoencoder(input_dim, encoding_dim):
    encoder = build_encoder(input_dim, encoding_dim)
    decoder = build_decoder(encoding_dim, input_dim)
    
    autoencoder_input = layers.Input(shape=(input_dim,))

    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    
    autoencoder = models.Model(autoencoder_input, decoded, name="autoencoder")
    return autoencoder, encoder, decoder