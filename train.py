import sys
import pickle
import numpy as np
from MemoryNetwork import MemoryNeuralNetwork
import time

save_path = "trained_models/trained_mnn_highd_5e4_100.obj"

with open("raw/data/three_track_train_data.npy", "rb") as f:
    np_train_data = np.load(f)

neeta=5e-4
neeta_dash=4e-5
lipschitz_constant = 1.0
epochs = 100

mnn = MemoryNeuralNetwork(2, 50, 2, neeta=neeta, neeta_dash=neeta_dash, lipschitz_norm=lipschitz_constant, spectral_norm=False)

actual_data = np.zeros((2, len(np_train_data)))
predicted_data = np.zeros((2, len(np_train_data)))

unstable_flag = False
error_sum = 0.0
error_list = np.zeros(epochs)

print("=====================================================================================================")
print("Training Started for %3d epochs, with the following parameters: \n eta: " % (epochs) + str(neeta) + "\n eta_dash: " + str(neeta_dash) + "\n lipschitz constant: " + str(lipschitz_constant) + " \n\n")


try:
    for _ in range(0, 2000):
        mnn.feedforward(np.zeros(2))
        mnn.backprop(np.zeros(2))

    for epoch in range(0, epochs):

        if(epoch != 0):
            print("Training for epoch %2d finished with average loss of %5.5f\n" % (epoch-1, error_sum/len(np_train_data)))
            error_list[epoch] = error_sum / len(np_train_data)
        error_sum = 0
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        for i in range(1, len(np_train_data)):
            
            if(mnn.squared_error > 1e30):
                
                unstable_flag = True
                #double break
                i = sys.maxint
                epoch = sys.maxint
                
            start_time = time.time()
            mnn.feedforward(np_train_data[i-1,:])
            end_time = time.time()
            print('run time: ', (end_time - start_time))
            print()
            mnn.backprop(np_train_data[i,:])
            
            error_sum += mnn.squared_error
            
            print("Training for epoch %2d, progress %5.2f%% with squared loss: %5.2f" % (epoch, (i/len(np_train_data)) * 100, mnn.squared_error), end="\r")
        #print()

    print("=====================================================================================================")    
    
except Exception as e:
    #print(e.__doc__)
    print(e)    
    
finally:
    if not unstable_flag:
        print("Done! saving model as " + save_path + " ...")
        
        filename = open(save_path, "wb")
        pickle.dump(mnn, filename)
        filename.close()
        '''
        plt.plot(error_list[1:])
        plt.xlabel("Epochs");plt.ylabel("Squared Loss");plt.title("Squared Loss vs Epochs")
        plt.grid(True)
        plt.savefig("trained_models/loss.png")
        plt.show()
        '''
    else:
        print("Network Unstable! Quitting...")

