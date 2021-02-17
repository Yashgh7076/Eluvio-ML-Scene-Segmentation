import pickle
import torch 
import os

directory = 'data'

files = os.listdir(directory)
count = 0

for i in files:
    #filename = os.path.join(directory,i)
    filename = directory + '/' + i
    #print(filename)
    with open(filename,'rb') as f:
        data = pickle.load(f)

    feat1 = data['place']
    feat1 = feat1.data.numpy() #convert tensors into numpy arrays for sklearn
    print(feat1.shape)

    feat2 = data['cast']
    feat2 = feat2.data.numpy()
    print(feat2.shape)

    feat3 = data['action']
    feat3 = feat3.data.numpy()
    print(feat3.shape)

    feat4 = data['audio']
    feat4 = feat4.data.numpy()
    print(feat4.shape)

    a = data['scene_transition_boundary_ground_truth']
    a = a.data.numpy()

    b = data['scene_transition_boundary_prediction']
    b = b.data.numpy()

    c = data['shot_end_frame']
    c = c.data.numpy()

    d = data['imdb_id']

    new_dict = dict([('scene_transition_boundary_ground_truth',a), ('scene_transition_boundary_prediction',b), ('shot_end_frame',c), ('imdb_id',d)])
    
    # use this routine to write final output pickle file
    #new_filename = d + '.pkl'
    #with open(new_filename, 'wb') as handle:
    #    pickle.dump(new_dict, handle)
    
