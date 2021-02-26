import pickle
import torch
import numpy as np
import sklearn
import os
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info
import tensorflow as tf
import matplotlib.pyplot as plt
from focal_loss import BinaryFocalLoss
import time

#get all files in the folder
directory = 'data'
files = os.listdir(directory) 
total_files = len(files) #Calculate total number of files 
windows_in_dataset = np.zeros((total_files), dtype = np.uint16) #since maximum number of shots in dataset is 3096

l = 0
window = 7 # Window needs to be an int greater than 1 and odd!
first = int((window - 1)/2)

start_master = time.time()
g = open('file_ID5.txt','a')
for i in range(50, 64):
    
    filename = directory + '/' + files[i]
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
        
    feat1 = data['place']
    feat1 = feat1.data.numpy() #convert tensors into numpy arrays for sklearn
    feat1_size = feat1.shape[1]
    
    feat2 = data['cast']
    feat2 = feat2.data.numpy()
    feat2_size = feat2.shape[1]
    
    feat3 = data['action']
    feat3 = feat3.data.numpy()
    feat3_size = feat3.shape[1]
    
    feat4 = data['audio']
    feat4 = feat4.data.numpy()
    feat4_size = feat4.shape[1]
    
    x = np.hstack((feat1, feat2, feat3, feat4))
    y = data['scene_transition_boundary_ground_truth']
    #y_new = y.data.numpy()
            
    scaler = preprocessing.MinMaxScaler().fit(x)
    x_scaled = scaler.transform(x)
 
    # Pad the start and end with zeros 
    padding = np.zeros((first, x_scaled.shape[1]))
    x_scaled = np.concatenate((padding, x_scaled, padding), axis=0)
                
    #Fold the data set to obtain features from adjoining shots
    N = x_scaled.shape[0] #changed from x_scaled
    j = 0
    GT = []
    start_s = time.time()

    for p in range(first, (N - first) - 1):
        #window_range = np.arange(start = p - first, stop = p + first + 1)
        temp1 = x_scaled[p - first: p + first + 1, :]
        #print(p - first, p + first + 1, p - first, temp1.shape[0], temp1.shape[1])
        temp1 = np.reshape(temp1, (1, window, temp1.shape[1]))
        
        temp2 = y[p - first].data.numpy()
        #print(p -first)
        temp2 = str(temp2)
        if(j == 0):
            X = temp1
        else:
            X = np.concatenate((X, temp1), axis=0)

        GT.append(temp2)
        j = j + 1    

    end_s = time.time()
    print('Iter ID:',i,' ','Time needed in s', end_s - start_s,' ','Array sizes, X:', X.shape,' ','GT:',len(GT))
    
    out_string = str(i) + ' ' + data['imdb_id'] + '.pkl'
    print(out_string)
    g.write(out_string)
    g.write('\n')

    if (l == 0):
        X_data = X
        Y_data = GT
    else:
        X_data = np.concatenate((X_data, X), axis = 0)
        Y_data.extend(GT)
           
    l = l + 1

g.close()
end_master = time.time()
print('X_data')
print('Final array size:', ' ', X_data.shape, ' ', len(Y_data), ' ', 'Total time needed in s:', end_master - start_master)

#convert ground truth predictions to integers 1->Scene boundary 0-> Not a scene boundary
M = len(Y_data)
#print('Y_gt')
Y_gt = np.zeros((M), dtype = np.uint8)
for i in range(M):
    if(Y_data[i] == 'True'):
        Y_gt[i] = 1
    elif(Y_data[i] == 'False'):
        Y_gt[i] = 0

# save array to file for easy load later
print('Saving array to file \n')
out_filename = 'dataset5' + '.npz'
print(out_filename)
np.savez_compressed(out_filename, a = X_data, b = Y_gt)   
