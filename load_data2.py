import pickle
import torch

with open('data/tt0052357.pkl','rb') as f:
    data = pickle.load(f)

print(type(data))