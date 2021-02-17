import pickle
import torch

with open('data/tt0052357.pkl','rb') as f:
    data = pickle.load(f)

print(type(data))

x = list(data)

print(x)

a = data['scene_transition_boundary_ground_truth']
a = a.data.numpy()
#print(a)
#print(a.shape)
#print(type(a))

b = data['scene_transition_boundary_prediction']
b = b.data.numpy()
#print(type(b))

c = data['shot_end_frame']
c = c.data.numpy()
#print(c)
#print(type(c))

d = data['imdb_id']
#print(d)
#print(type(d))

new_dict = dict([('scene_transition_boundary_ground_truth', a), ('scene_transition_boundary_prediction',b), ('shot_end_frame',c), ('imdb_id',d)])
print(list(new_dict))

#new_filename = d + '.pkl'
#with open(new_filename, 'wb') as handle:
#    pickle.dump(new_dict, handle)

