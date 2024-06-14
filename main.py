from data import load_data_sets
from keras.optimizers import Adam
from models import build_autoencoder, build_decoder, build_encoder
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main(model_name="base",epoch_count=50,batch_size=256, X_train=[], X_test=[], encoding_dimension=32, encoding_layers=[]):
    
    Path("saved_plots/"+model_name).mkdir(parents=True, exist_ok=True)
    Path("models/"+model_name).mkdir(parents=True, exist_ok=True)

    if len(X_train)==0 or len(X_test)==0:
        X_train, X_test = load_data_sets()

    # print(X_train.shape[1])
    # print(X_test.shape[1])

    X_train = X_train[:2000]
    X_test = X_test[:2000]

    input_dim = X_train.shape[1]
    encoding_dim = encoding_dimension 

    if len(encoding_layers)==0:
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
                    epochs=epoch_count,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    encoded_data = encoder.predict(X_test)
    decoded_data = decoder.predict(encoded_data)
    
    mse = np.mean(np.square(X_test - decoded_data))
    print("Mean Squared Error:", mse)

    mae = np.mean(np.abs(X_test - decoded_data))
    print("Mean Absolute Error:", mae)

    encoder.save("models/"+model_name+"/"+model_name+"_encoder.keras")
    decoder.save("models/"+model_name+"/"+model_name+"_decoder.keras")

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(str("saved_plots/"+model_name+"/Loss.png"), dpi=600)
    
    f=open("saved_plots/"+model_name+"/model_"+model_name+"_data","w")
    f.write(
        "Model name: "+str(model_name)+
        "\nEpochs: "+str(epoch_count)+
        "\nbatch size: "+str(batch_size)+
        "\nencoding dimension:"+str(encoding_dimension)+
        "\nMean Squared Error:"+ str(mse)+
        "\nMean Absolute Error:"+str(mae)+
        "\nlayers: "+str(encoding_layers))
    f.close() 

    # plt.show()

if __name__ == '__main__':
    main("base")