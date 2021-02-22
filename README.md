This is a repository to store files while I work on the Eluvio ML Scene Segmentation Challenge

Update #1:
1) 16th February 2021: Successfully wrote code to read and store features, and output pickle file for all 64 movies.

Update #2:

Description for each neural network model attempted

1) cnn.py -> Create copies of 'cast', 'action', 'audio' to create a feature vector of size (2048, 4) for each pair of shots that define a boundary. Then a Siamese CNN is used to learn features from adjoining shots and classify shot boundary as scene boundary or not.

2) dense.py -> Learn a joint dense network on the same data as 1)

3) siamese.py -> Learn a set of CNN each for 'places','cast','action' and 'audio' and combine the activation maps before transferring to a dense network.

4) siamese_2.py -> Learn 2 CNN's for the shots the define the boundary.

5) siamese_3_class_imbalance.py -> Same as siamese_2.py but use BinaryFocalLoss as loss instead of BinaryCrossEntropy

6) simple_cnn.py -> Learn two CNN's for adjoining shots and combine activation map before feeding into a dense network.

7) time_series.py -> Reformat the data such that an individual data frame is of size (window, 3584) where window can be flexibly chosen. This approach considers a sliding window approach to classification by allowing a CNN to see the neighboring shots of a boundary.

Motivation for using Focal Loss as defined in https://arxiv.org/abs/1708.02002 and https://pypi.org/project/focal-loss/:
On observing the data it is seen that there are close to 8000 true scene boundaries and close to 97000 ordinary shot boundaries. This is an example of an imbalanced data set for a one-shot detector that was being developed. Focal loss has been demonstrated to help in such situations and the code is conveniently developed in a Python package.
