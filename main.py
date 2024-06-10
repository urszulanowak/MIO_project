from data import load_data_sets
from keras.optimizers import Adam
from models import build_autoencoder
import matplotlib.pyplot as plt

def main():
    X_train, X_test = load_data_sets()
    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train[:2000]
    X_test = X_test[:2000]

    input_dim = X_train.shape[1]
    encoding_dim = 32 
    autoencoder, encoder, decoder = build_autoencoder(input_dim, encoding_dim)

    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
    history =  autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    encoded_data = encoder.predict(X_test)
    decoded_data = decoder.predict(encoded_data)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
if __name__ == '__main__':
    main()