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
            
    scaler = preprocessing.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
        
    #Fold the data set to obtain features from adjoining shots
    N = x.shape[0] #changed from x_scaled
    shots_per_movie[k] = N
    x_fold = np.zeros((N - 1, 2*x.shape[1])) #changed from x_scaled
    for l in range(N - 1):
        x_fold[l,:] = np.hstack((x_scaled[l,:], x_scaled[l+1,:])) #changed from x_scaled
        
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
x1, x2, x3, x4, x5, x6, x7, x8 = tf.split(inputs, array_dims, axis = 1) # split inputs into given features for two consecutive shots

x1_r = tf.keras.layers.Reshape((feat1_size, 1))(x1)
#x1_b = tf.keras.layers.BatchNormalization(axis=1)(x1_r)

x5_r = tf.keras.layers.Reshape((feat1_size, 1))(x5)
#x5_b = tf.keras.layers.BatchNormalization(axis=1)(x5_r)

#x2b = tf.keras.layers.BatchNormalization(axis=1)(x2)
x2_conc = tf.keras.layers.concatenate([x2, x2, x2, x2], axis = 1) # create vectors of size (None, 2048) from size (None, 512)
x2_r = tf.keras.layers.Reshape((feat1_size, 1))(x2_conc)
print(x2_r.shape)

#x3b = tf.keras.layers.BatchNormalization(axis=1)(x3)
x3_conc = tf.keras.layers.concatenate([x3, x3, x3, x3], axis = 1)
x3_r = tf.keras.layers.Reshape((feat1_size, 1))(x3_conc)
print(x3_r.shape)

#x4b = tf.keras.layers.BatchNormalization(axis=1)(x4)
x4_conc = tf.keras.layers.concatenate([x4, x4, x4, x4], axis = 1)
x4_r = tf.keras.layers.Reshape((feat1_size, 1))(x4_conc)
print(x4_r.shape)

#x6b = tf.keras.layers.BatchNormalization(axis=1)(x6)
x6_conc = tf.keras.layers.concatenate([x6, x6, x6, x6], axis = 1) # create vectors of size (None, 2048) from size (None, 512)
x6_r = tf.keras.layers.Reshape((feat1_size, 1))(x6_conc)
print(x6_r.shape)

#x7b = tf.keras.layers.BatchNormalization(axis=1)(x7)
x7_conc = tf.keras.layers.concatenate([x7, x7, x7, x7], axis = 1)
x7_r = tf.keras.layers.Reshape((feat1_size, 1))(x7_conc)
print(x7_r.shape)

#x8b = tf.keras.layers.BatchNormalization(axis=1)(x8)
x8_conc = tf.keras.layers.concatenate([x8, x8, x8, x8], axis = 1)
x8_r = tf.keras.layers.Reshape((feat1_size, 1))(x8_conc)
print(x8_r.shape)

shot_1 = tf.keras.layers.concatenate([x1_r, x2_r, x3_r, x4_r], axis = 2) # create a vector of size (None, 2048, 4)
print(shot_1.shape)

shot_2 = tf.keras.layers.concatenate([x5_r, x6_r, x7_r, x8_r], axis = 2) # create a vector of size (None, 2048, 4)
print(shot_2.shape)

# Convolutional block 1
conv1 = tf.keras.Sequential()
conv1.add(tf.keras.layers.Reshape((1, feat1_size, 4)))

conv1.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(1, 15), strides=1, padding='same', activation='linear')) #input size (None, 1, 2048, 4)
conv1.add(tf.keras.layers.BatchNormalization(axis=3))
conv1.add(tf.keras.layers.ReLU())
conv1.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 1024, 6)

conv1.add(tf.keras.layers.Conv2D(filters=12, kernel_size=(1, 15), strides=1, padding='same', activation='linear')) #input size (None, 1, 1024, 6)
conv1.add(tf.keras.layers.BatchNormalization(axis=3))
conv1.add(tf.keras.layers.ReLU())
conv1.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 512, 12)

conv1.add(tf.keras.layers.Conv2D(filters=18, kernel_size=(1, 7), strides=1, padding='same', activation='linear')) #input size (None, 1, 512, 12)
conv1.add(tf.keras.layers.BatchNormalization(axis=3))
conv1.add(tf.keras.layers.ReLU())
conv1.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 256, 18)

conv1.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 7), strides=1, padding='same', activation='linear')) #input size (None, 1, 256, 18)
conv1.add(tf.keras.layers.BatchNormalization(axis=3))
conv1.add(tf.keras.layers.ReLU())
conv1.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 128, 24)

conv1.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 7), strides=1, padding='same', activation='linear')) #input size (None, 1, 128, 24)
conv1.add(tf.keras.layers.BatchNormalization(axis=3))
conv1.add(tf.keras.layers.ReLU())
conv1.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 64, 24)

conv1.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 3), strides=1, padding='same', activation='linear')) #input size (None, 1, 64, 24)
conv1.add(tf.keras.layers.BatchNormalization(axis=3))
conv1.add(tf.keras.layers.ReLU())
conv1.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 32, 24)

# Convolutional block 2
conv2 = tf.keras.Sequential()
conv2.add(tf.keras.layers.Reshape((1, feat1_size, 4)))

conv2.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(1, 15), strides=1, padding='same', activation='linear')) #input size (None, 1, 2048, 4)
conv2.add(tf.keras.layers.BatchNormalization(axis=3))
conv2.add(tf.keras.layers.ReLU())
conv2.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 1024, 6)

conv2.add(tf.keras.layers.Conv2D(filters=12, kernel_size=(1, 15), strides=1, padding='same', activation='linear')) #input size (None, 1, 1024, 6)
conv2.add(tf.keras.layers.BatchNormalization(axis=3))
conv2.add(tf.keras.layers.ReLU())
conv2.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 512, 12)

conv2.add(tf.keras.layers.Conv2D(filters=18, kernel_size=(1, 7), strides=1, padding='same', activation='linear')) #input size (None, 1, 512, 12)
conv2.add(tf.keras.layers.BatchNormalization(axis=3))
conv2.add(tf.keras.layers.ReLU())
conv2.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 256, 18)

conv2.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 7), strides=1, padding='same', activation='linear')) #input size (None, 1, 256, 18)
conv2.add(tf.keras.layers.BatchNormalization(axis=3))
conv2.add(tf.keras.layers.ReLU())
conv2.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 128, 24)

conv2.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 7), strides=1, padding='same', activation='linear')) #input size (None, 1, 128, 24)
conv2.add(tf.keras.layers.BatchNormalization(axis=3))
conv2.add(tf.keras.layers.ReLU())
conv2.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 64, 24)

conv2.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 3), strides=1, padding='same', activation='linear')) #input size (None, 1, 64, 24)
conv2.add(tf.keras.layers.BatchNormalization(axis=3))
conv2.add(tf.keras.layers.ReLU())
conv2.add(tf.keras.layers.MaxPool2D(pool_size=(1,2))) #output size (None, 1, 32, 24)

encoded_left = conv1(shot_1)
encoded_right = conv2(shot_2)
combine = tf.keras.layers.concatenate([encoded_left, encoded_right], axis = 2)
flat = tf.keras.layers.Flatten()(combine) #output size (None, 1536)
d1 = tf.keras.layers.Dropout(rate=0.5)(flat)
d2 = tf.keras.layers.Dense(50, activation='relu')(d1)
d2r = tf.keras.layers.Dropout(rate=0.5)(d2)
d3 = tf.keras.layers.Dense(25, activation='relu')(d2r)
d3r = tf.keras.layers.Dropout(rate=0.5)(d3)
output = tf.keras.layers.Dense(1, activation = 'softmax')(d3r)

model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(optimizer = 'adam', loss = BinaryFocalLoss(pos_weight = 100, gamma=2.5), metrics=[tf.keras.metrics.Accuracy()])
model.summary()

tf.keras.utils.plot_model(model, to_file='simple_cnn_FL128.png', dpi=100)

training_log = 'crossval_fold_' + '.txt'
csv_logger = tf.keras.callbacks.CSVLogger(training_log, append = True, separator=' ')
metrics = model.fit(X, Y_gt, epochs=100, validation_split= 0.2, verbose=2, batch_size = 128)