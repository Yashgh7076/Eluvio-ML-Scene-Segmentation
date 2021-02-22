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

#get all files in the folder
directory = 'data'
files = os.listdir(directory) 
total_files = len(files) #Calculate total number of files 
windows_in_dataset = np.zeros((total_files), dtype = np.uint16) #since maximum number of shots in dataset is 3096

l = 0
m = 0
window = 5 # Window needs to be an int greater than 1 and odd!
first = int((window - 1)/2)
previous_windows = 0

for i in range(total_files):
    filename = directory + '/' + files[i]
    print(filename)
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
            
    scaler = preprocessing.MinMaxScaler().fit(x)
    x_scaled = scaler.transform(x)
        
    #Fold the data set to obtain features from adjoining shots
    N = x.shape[0] #changed from x_scaled
    j = 0
    GT = []
    for p in range(first, (N - first) - 1):
        #window_range = np.arange(start = p - first, stop = p + first + 1)
        #print(p-first, p+first+1, p)
        temp1 = x_scaled[p - first: p + first + 1, :]
        temp1 = np.reshape(temp1, (1, window, temp1.shape[1]))

        temp2 = y[p].data.numpy()
        temp2 = str(temp2)
        if(j == 0):
            X = temp1
        else:
            X = np.concatenate((X, temp1), axis=0)

        GT.append(temp2)
        j = j + 1    

    print('X')        
    print(X.shape)
    print(len(GT))
    windows_in_dataset[m] = len(GT) 
    
    if (l == 0):
        X_data = X
        Y_data = GT
    else:
        X_data = np.concatenate((X_data, X), axis = 0)
        Y_data.extend(GT)
           
    l = l + 1
    m = m + 1

print('X_data')
print(windows_in_dataset)
print(X_data.shape)
print(len(Y_data))
#print('Y_data')
#for i in range(len(Y_data)):
#    print(Y_data[i])

#convert ground truth predictions to integers 1->Scene boundary 0-> Not a scene boundary
M = len(Y_data)
#print('Y_gt')
Y_gt = np.zeros((M), dtype = np.uint8)
for i in range(M):
    if(Y_data[i] == 'True'):
        Y_gt[i] = 1
        #print('1')
    elif(Y_data[i] == 'False'):
        Y_gt[i] = 0
        #print('0')
#print(Y_gt)

# Calculate values for cross-validation later
iter_sizes = np.zeros((4), dtype = np.uint32)
iter_sizes[0] = 0.2*total_files - 1
iter_sizes[1] = 0.4*total_files - 1
iter_sizes[2] = 0.6*total_files - 1
iter_sizes[3] = 0.8*total_files - 1

iter_ids = np.zeros((5,2), dtype = np.uint32)
iter_ids[0,0] = 0 
iter_ids[0,1] = windows_in_dataset[iter_sizes[0]]

iter_ids[1,0] = windows_in_dataset[iter_sizes[0]]
iter_ids[1,1] = windows_in_dataset[iter_sizes[1]]

iter_ids[2,0] = windows_in_dataset[iter_sizes[1]]
iter_ids[2,1] = windows_in_dataset[iter_sizes[2]]

iter_ids[3,0] = windows_in_dataset[iter_sizes[2]]
iter_ids[3,1] = windows_in_dataset[iter_sizes[3]]

iter_ids[4,0] = windows_in_dataset[iter_sizes[3]]
iter_ids[4,1] = len(Y_data)
print(iter_ids)

# define keras model from here
inputs = tf.keras.layers.Input(shape = (X_data.shape[1], X_data.shape[2]))

conv = tf.keras.Sequential()
conv.add(tf.keras.layers.Conv1D(filters = 8, kernel_size = 3, strides = 1, padding='same', activation='linear')) #input size (None, 5, 3584)
conv.add(tf.keras.layers.BatchNormalization(axis=2))
conv.add(tf.keras.layers.ReLU())
conv.add(tf.keras.layers.MaxPool1D(pool_size = 2)) #output size (None, 2, 8)

encode = conv(inputs)
flat = tf.keras.layers.Flatten()(encode) #output size (None, 10)

#d2 = tf.keras.layers.Dense(50, activation='relu')(flat)
#d2r = tf.keras.layers.Dropout(rate=0.5)(d2)
#d2d = tf.keras.layers.BatchNormalization(axis=1)(d2r)

#d3 = tf.keras.layers.Dense(12, activation='relu')(flat)
#d3r = tf.keras.layers.Dropout(rate=0.5)(d3)
#d3d = tf.keras.layers.BatchNormalization(axis=1)(d3)

output = tf.keras.layers.Dense(1, activation = 'softmax')(flat)

model = tf.keras.Model(inputs = inputs, outputs = output)
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(optimizer = opt, loss = BinaryFocalLoss(pos_weight = 9, gamma=2.5), metrics=[tf.keras.metrics.Accuracy()])
model.summary()

#tf.keras.utils.plot_model(model, to_file='CNN1D_FL128.png', dpi=100)

#training_log = 'crossval_fold_' + '.txt'
#csv_logger = tf.keras.callbacks.CSVLogger(training_log, append = True, separator=' ')
metrics = model.fit(X_data, Y_gt, epochs=100, validation_split= 0.2, verbose=2, batch_size = 32)







