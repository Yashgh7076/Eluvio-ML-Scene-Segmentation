import pickle
import torch
import numpy as np
import sklearn
import os
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info
import tensorflow as tf
import matplotlib.pyplot as plt

#get all files in the folder
directory = 'data'
files = os.listdir(directory) 
total_files = len(files) #Calculate total number of files 
shots_per_movie = np.zeros((total_files), dtype = np.uint16) #since maximum number of shots in dataset is 3096

k = 0
j = 0
for i in files:
    filename = directory + '/' + i
    with open(filename,'rb') as f:
        data = pickle.load(f)
        
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
            
    #scaler = preprocessing.StandardScaler().fit(x)
    #x_scaled = scaler.transform(x)
        
    #Fold the data set to obtain features from adjoining shots
    N = x.shape[0] #changed from x_scaled
    shots_per_movie[k] = N
    x_fold = np.zeros((N - 1, 2*x.shape[1])) #changed from x_scaled
    for i in range(N - 1):
        x_fold[i,:] = np.hstack((x[i,:], x[i+1,:])) #changed from x_scaled
        
    if (j == 0):
        X = x_fold
        Y = y
    else:
        X = np.concatenate((X, x_fold), axis = 0)
        Y = np.concatenate((Y, y))
        
    j = j + 1
    k = k + 1
    
#print(X)
#print(X.shape)

#print(Y)
#print(Y.shape)  
   
#print(shots_per_movie)
#print(shots_per_movie.shape)

#convert ground truth predictions to integers 1->Scene boundary 0-> Not a scene boundary
M = Y.shape[0]
Y_gt = np.zeros((M), dtype = np.uint8)
for i in range(M):
    if(Y[i] == True):
        Y_gt[i] = 1
    elif(Y[i] == False):
        Y_gt[i] = 0
#print(Y_gt)

# Create train and test datasets for cross-validation
shots_in_dataset = np.zeros((total_files), dtype = np.uint32)
previous_shots = 0
for i in range(total_files):
    shots_per_movie[i] = shots_per_movie[i] - 1
    shots_in_dataset[i] = shots_per_movie[i] + previous_shots
    previous_shots = shots_in_dataset[i]
    
#print(shots_per_movie)
#print(shots_in_dataset)

iter_sizes = np.zeros((4), dtype = np.uint32)
iter_sizes[0] = 0.2*total_files - 1
iter_sizes[1] = 0.4*total_files - 1
iter_sizes[2] = 0.6*total_files - 1
iter_sizes[3] = 0.8*total_files - 1
    
# Create IDs for cross-validation
iter_ids = np.zeros((5,2), dtype = np.uint32)
iter_ids[0,0] = 0 
iter_ids[0,1] = shots_in_dataset[iter_sizes[0]]

iter_ids[1,0] = shots_in_dataset[iter_sizes[0]]
iter_ids[1,1] = shots_in_dataset[iter_sizes[1]]

iter_ids[2,0] = shots_in_dataset[iter_sizes[1]]
iter_ids[2,1] = shots_in_dataset[iter_sizes[2]]

iter_ids[3,0] = shots_in_dataset[iter_sizes[2]]
iter_ids[3,1] = shots_in_dataset[iter_sizes[3]]

iter_ids[4,0] = shots_in_dataset[iter_sizes[3]]
iter_ids[4,1] = M
#print(iter_ids)

array_dims = np.zeros((8), dtype = np.int32)
array_dims[0] = feat1_size
array_dims[1] = feat2_size #feat1_size + feat2_size
array_dims[2] = feat3_size #feat1_size + feat2_size + feat3_size
array_dims[3] = feat4_size #feat1_size + feat2_size + feat3_size + feat4_size
array_dims[4] = array_dims[0] #array_dims[3] + feat1_size
array_dims[5] = array_dims[1] #array_dims[3] + feat1_size + feat2_size
array_dims[6] = array_dims[2] #array_dims[3] + feat1_size + feat2_size + feat3_size
array_dims[7] = array_dims[3] #array_dims[3] + feat1_size + feat2_size + feat3_size + feat4_size
#print(array_dims)

#labels = Y_gt[0:100]

#define model using keras functional API
inputs = tf.keras.layers.Input(shape = (X.shape[1]))
x1, x2, x3, x4, x5, x6, x7, x8 = tf.split(inputs, array_dims, axis = 1)

# First siamese model for processing places information
siamese_model1 = tf.keras.Sequential()
siamese_model1.add(tf.keras.layers.Reshape((feat1_size, 1))) 
siamese_model1.add(tf.keras.layers.BatchNormalization(axis = 1))

siamese_model1.add(tf.keras.layers.Conv1D(filters = 3, kernel_size = 15, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 2048, 1)
siamese_model1.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model1.add(tf.keras.layers.ReLU())
siamese_model1.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model1.add(tf.keras.layers.Conv1D(filters = 6, kernel_size = 15, strides = 1, padding = 'same', activation = 'linear')) #input size (None, 1024, 3)
siamese_model1.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model1.add(tf.keras.layers.ReLU())
siamese_model1.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model1.add(tf.keras.layers.Conv1D(filters = 12, kernel_size = 15, strides = 1, padding = 'same', activation = 'linear')) #input size (None, 512, 6)
siamese_model1.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model1.add(tf.keras.layers.ReLU())
siamese_model1.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model1.add(tf.keras.layers.Conv1D(filters = 25, kernel_size = 7, strides = 1, padding = 'same', activation = 'linear')) #input size (None, 256, 12)
siamese_model1.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model1.add(tf.keras.layers.ReLU())
siamese_model1.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model1.add(tf.keras.layers.Conv1D(filters = 50, kernel_size = 7, strides = 1, padding = 'same', activation = 'linear')) #input size (None, 128, 25)
siamese_model1.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model1.add(tf.keras.layers.ReLU())
siamese_model1.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model1.add(tf.keras.layers.Conv1D(filters = 100, kernel_size = 3, strides = 1, padding = 'same', activation = 'linear')) #input size (None, 64, 50)
siamese_model1.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model1.add(tf.keras.layers.ReLU())
siamese_model1.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last')) #output size (None, 32, 100)

# Second siamese model for processing cast information
siamese_model2 = tf.keras.Sequential()
siamese_model2.add(tf.keras.layers.Reshape((feat2_size, 1))) 
siamese_model2.add(tf.keras.layers.BatchNormalization(axis = 1))

siamese_model2.add(tf.keras.layers.Conv1D(filters = 6, kernel_size = 15, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 512, 1)
siamese_model2.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model2.add(tf.keras.layers.ReLU())
siamese_model2.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model2.add(tf.keras.layers.Conv1D(filters = 12, kernel_size = 7, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 256, 6)
siamese_model2.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model2.add(tf.keras.layers.ReLU())
siamese_model2.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model2.add(tf.keras.layers.Conv1D(filters = 25, kernel_size = 7, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 128, 12)
siamese_model2.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model2.add(tf.keras.layers.ReLU())
siamese_model2.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model2.add(tf.keras.layers.Conv1D(filters = 50, kernel_size = 3, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 64, 25)
siamese_model2.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model2.add(tf.keras.layers.ReLU())
siamese_model2.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last')) # output size (None, 32, 50)

#siamese model 3 for processing action information
siamese_model3 = tf.keras.Sequential()
siamese_model3.add(tf.keras.layers.Reshape((feat3_size, 1))) 
siamese_model3.add(tf.keras.layers.BatchNormalization(axis = 1))

siamese_model3.add(tf.keras.layers.Conv1D(filters = 6, kernel_size = 15, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 512, 1)
siamese_model3.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model3.add(tf.keras.layers.ReLU())
siamese_model3.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model3.add(tf.keras.layers.Conv1D(filters = 12, kernel_size = 7, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 256, 6)
siamese_model3.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model3.add(tf.keras.layers.ReLU())
siamese_model3.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model3.add(tf.keras.layers.Conv1D(filters = 25, kernel_size = 7, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 128, 12)
siamese_model3.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model3.add(tf.keras.layers.ReLU())
siamese_model3.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model3.add(tf.keras.layers.Conv1D(filters = 50, kernel_size = 3, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 64, 25)
siamese_model3.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model3.add(tf.keras.layers.ReLU())
siamese_model3.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last')) # output size (None, 32, 50)

#siamese model 4 for processing audio information
siamese_model4 = tf.keras.Sequential()
siamese_model4.add(tf.keras.layers.Reshape((feat4_size, 1))) 
siamese_model4.add(tf.keras.layers.BatchNormalization(axis = 1))

siamese_model4.add(tf.keras.layers.Conv1D(filters = 6, kernel_size = 15, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 512, 1)
siamese_model4.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model4.add(tf.keras.layers.ReLU())
siamese_model4.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model4.add(tf.keras.layers.Conv1D(filters = 12, kernel_size = 7, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 256, 6)
siamese_model4.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model4.add(tf.keras.layers.ReLU())
siamese_model4.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model4.add(tf.keras.layers.Conv1D(filters = 25, kernel_size = 7, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 128, 12)
siamese_model4.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model4.add(tf.keras.layers.ReLU())
siamese_model4.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last'))

siamese_model4.add(tf.keras.layers.Conv1D(filters = 50, kernel_size = 3, strides = 1, padding = 'same', activation = 'linear'))#input size (None, 64, 25)
siamese_model4.add(tf.keras.layers.BatchNormalization(axis = 2))
siamese_model4.add(tf.keras.layers.ReLU())
siamese_model4.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, data_format = 'channels_last')) # output size (None, 32, 50)

# pass information of Left and Right shot through Siamese architecture
el1 = siamese_model1(x1)
er1 = siamese_model1(x5)

el2 = siamese_model2(x2)
er2 = siamese_model2(x6)

el3 = siamese_model3(x3)
er3 = siamese_model3(x7)

el4 = siamese_model4(x4)
er4 = siamese_model4(x8)

combine = tf.keras.layers.Concatenate(axis=2)([el1, er1, el2, er2, el3, er3, el4, er4]) #output size (None, 32, 500)
conv1 = tf.keras.layers.Conv1D(filters=200, kernel_size=1, strides = 1, padding = 'same', activation = 'linear')(combine)
bn1 = tf.keras.layers.BatchNormalization(axis = 2)(conv1)
act1 = tf.keras.layers.ReLU()(bn1)
out1 = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, padding = 'same', data_format = 'channels_last')(act1) #output size (None, 16, 200)

conv2 = tf.keras.layers.Conv1D(filters=100, kernel_size=1, strides = 1, padding = 'same', activation = 'linear')(out1)
bn2 = tf.keras.layers.BatchNormalization(axis = 2)(conv2)
act2 = tf.keras.layers.ReLU()(bn2)
out2 = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2, padding = 'same', data_format = 'channels_last')(act2) #output size (None, 8, 100)

d1 = tf.keras.layers.Flatten()(out1) #output size (None, 800)
d2 = tf.keras.layers.Dense(200, activation='relu')(d1)
d2r = tf.keras.layers.Dropout(rate=0.5)(d2)
d3 = tf.keras.layers.Dense(100, activation='relu')(d2r)
d3r = tf.keras.layers.Dropout(rate=0.5)(d3)
output = tf.keras.layers.Dense(1, activation = 'softmax')(d3r)

model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits='True'), metrics=[tf.keras.losses.BinaryCrossentropy(from_logits='True')])
model.summary()

tf.keras.utils.plot_model(model, to_file='siamese_model.png', dpi=100)

#training_log = 'crossval_fold_' + '.txt'
#csv_logger = tf.keras.callbacks.CSVLogger(training_log, append = True, separator=' ')
#metrics = model.fit(X, Y_gt, epochs=10, validation_split= 0.2, verbose=2)