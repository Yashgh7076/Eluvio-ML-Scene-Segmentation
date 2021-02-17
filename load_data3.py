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

    a = data['scene_transition_boundary_ground_truth']
    a = a.data.numpy()

    b = data['scene_transition_boundary_prediction']
    b = b.data.numpy()

    c = data['shot_end_frame']
    c = c.data.numpy()

    d = data['imdb_id']

    new_dict = dict([('scene_transition_boundary_ground_truth',a), ('scene_transition_boundary_prediction',b), ('shot_end_frame',c), ('imdb_id',d)])
    
    new_filename = d + '.pkl'
    with open(new_filename, 'wb') as handle:
        pickle.dump(new_dict, handle)
    
