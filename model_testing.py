from data import load_data_sets
from keras.optimizers import Adam
from models import build_autoencoder, build_decoder, build_encoder
import matplotlib.pyplot as plt
import numpy as np
import keras
import statistics
from pathlib import Path

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]


def model_testing(model_name :str, test_set):

    Path("saved_plots/"+model_name).mkdir(parents=True, exist_ok=True)
    
    f=open("saved_plots/"+model_name+"/model_"+model_name+"_data","a")
      
    encoder=keras.models.load_model("models/"+model_name+"/"+model_name+"_encoder.keras")
    decoder=keras.models.load_model("models/"+model_name+"/"+model_name+"_decoder.keras")

    # _, test_set = load_data_sets()
    # test_set=test_set[2000:]

    encoded_test = encoder.predict(test_set)
    decoded_test = decoder.predict(encoded_test)

    
    f.write("\nData:\n")
    
    f.write(f"Testing on {len(test_set)} records\n")
    f.write(f"Each record contains {len(test_set[0])} points\n")

    f.write("\nErrors:\n")
    f.write("For entire set:\n")
    mse = np.mean(np.square(test_set - decoded_test))
    f.write("Mean Squared Error:"+str(mse)+"\n")

    mae = np.mean(np.abs(test_set - decoded_test))
    f.write("Mean Absolute Error:"+str(mae)+"\n")

    mse_array=[]
    mae_array=[]
    # for index in range(len(test_set)):
    for index, test_value in enumerate(test_set):
        mse = np.mean(np.square(test_value - decoded_test[index]))
        mse_array.append(mse)
        mae = np.mean(np.abs(test_value - decoded_test[index]))
        mae_array.append(mae)


    ##############################
    #ploting
    ##############################

    #wykres błędów mse
    fig, (mae_plt, mse_plt)=plt.subplots(2,  figsize=(12, 6))
    fig.subplots_adjust(left=0.065, bottom=0.09)

    fig.suptitle("Errors values")
    mae_plt.plot(mae_array,linestyle="none",marker='.')
    mse_plt.plot(mse_array,linestyle="none",marker='.')
    mae_plt.set_title("mae")
    mse_plt.set_title("mse")
    # fig.tight_layout()

    mae_plt.set(xlabel='index',ylabel='mae')
    mse_plt.set(xlabel='index',ylabel='mse')
    mae_plt.label_outer()
    mse_plt.label_outer()
    fig.align_labels()
    plt.savefig(str("saved_plots/"+model_name+"/Error_values.png"), dpi=600)

    #indeksy kluczowe
    
    f.write("\nKey values:\n")
    #max mse
    ind_max=mse_array.index(max(mse_array))
    f.write(f"mse_max=mse[{ind_max}]={mse_array[ind_max]}\n")
    #min mse
    ind_min=mse_array.index(min(mse_array))
    f.write(f"mse_min=mse[{ind_min}]={mse_array[ind_min]}\n")
    #mid mse
    ind_mid=mse_array.index(closest(mse_array,(mse_array[ind_max]-mse_array[ind_min])/2))
    f.write(f"mse_mid=mse[{ind_mid}]={mse_array[ind_mid]}\n")
    #med mse
    ind_med=mse_array.index(closest(mse_array,statistics.median(mse_array)))
    f.write(f"mse_med=mse[{ind_med}]={mse_array[ind_med]}\n")

    #wykresy dla indeksów kluczowych
    fig2, plot_axis=plt.subplots(3,4, figsize=(12, 6))

    #plot y=0 oryginal
    plot_axis[0,0].plot(test_set[ind_min])
    plot_axis[0,1].plot(test_set[ind_med])
    plot_axis[0,2].plot(test_set[ind_mid])
    plot_axis[0,3].plot(test_set[ind_max])

    #titles
    plot_axis[0,0].set_title(f"mse_min\nmse[{ind_min}]={round(mse_array[ind_min],4)}",fontsize=10)
    plot_axis[0,1].set_title(f"mse_med\nmse[{ind_med}]={round(mse_array[ind_med],4)}",fontsize=10)
    plot_axis[0,2].set_title(f"mse_mid\nmse[{ind_mid}]={round(mse_array[ind_mid],4)}",fontsize=10)
    plot_axis[0,3].set_title(f"mse_max\nmse[{ind_max}]={round(mse_array[ind_max],4)}",fontsize=10)

    #plot y=1 coded
    plot_axis[1,0].plot(encoded_test[ind_min])
    plot_axis[1,1].plot(encoded_test[ind_med])
    plot_axis[1,2].plot(encoded_test[ind_mid])
    plot_axis[1,3].plot(encoded_test[ind_max])

    #plot y=2 decoded
    plot_axis[2,0].plot(decoded_test[ind_min])
    plot_axis[2,1].plot(decoded_test[ind_med])
    plot_axis[2,2].plot(decoded_test[ind_mid])
    plot_axis[2,3].plot(decoded_test[ind_max])

    #left labels
    plot_axis[0,0].set(ylabel='orginal')  
    plot_axis[1,0].set(ylabel='encoded') 
    plot_axis[2,0].set(ylabel='decoded')

    fig2.align_labels()
    plt.savefig("saved_plots/"+model_name+"/Key_values.png", dpi=600)
    f.close()

