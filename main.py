from data import load_data_sets
from keras.optimizers import Adam
from models import build_autoencoder, build_decoder, build_encoder
import matplotlib.pyplot as plt
import numpy as np

def main():
    X_train, X_test = load_data_sets()
    print(X_train.shape[1])
    print(X_test.shape[1])

    X_train = X_train[:2000]
    X_test = X_test[:2000]

    input_dim = X_train.shape[1]
    encoding_dim = 32 

    encoding_layers = [
        (64, 'relu'),
        (encoding_dim, 'relu')
    ]

    decoding_layers = [
        (64, 'relu'),
        (input_dim, 'sigmoid')
    ]

    encoder = build_encoder((input_dim,), encoding_layers)
    decoder = build_decoder((encoding_dim,), decoding_layers)
    autoencoder = build_autoencoder(input_dim, encoder, decoder)

    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
    history =  autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    encoded_data = encoder.predict(X_test)
    decoded_data = decoder.predict(encoded_data)
    
    mse = np.mean(np.square(X_test - decoded_data))
    print("Mean Squared Error:", mse)

    mae = np.mean(np.abs(X_test - decoded_data))
    print("Mean Absolute Error:", mae)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()