IMPORTANT!
Please see the folder final for the latest submissions for this challenge. Updates are posted regularly and can be found at the bottom of this README


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

Update #3: Date 25th Feb 2021

Obtained metrics Mean Average Precision (mAP) = 0.081, Mean Maximum IoU (miou) = 0.044

The idea was to use a window of shots to generate feature embeddings for the boundary after the central shot. So for examples shots 1 through 7 (assuming 0 index is first) would be used to learn embeddings to represent the boundary after shot 4. 

The main aspects of this approach are:

1) Since this is a sliding window approach the ends of each movie were padded with shots having 0 (zero padding). The number of shots was flexibly varied and is calculated as int((window-1)/2). It is important to note that the size of the window needs to be an odd integer greater than 1 so that each central shot has the same number of neighbors on either side. 

2) In order to have the same dimension for all features from 'place','cast','action' and 'audio', the latter three were repeated along their respective dimension and concatenated with the former. This means that each shots is represented as a feature vector of size 2048 x 4.

3) After reshaping, the final input to the neural network is of size (None, window, 2048, 4) where None is the batch size and can be varied flexibly as well. This essentially treats each input as an image of size window x 2048 having 4 channels of information given by the extracted features. This idea was explored in my Thesis seen here: https://yashgh7076.github.io/projects.html

4) Global Average Pooling was used to retain the average of all activation maps in the final convolutional layer. This was done primarily to keep the number of connections required to connect to the dense network down, which also helps in reducing the number of free parameters of the network.

5) Due to hardware limitations the dataset was divided into two parts having roughly 60% of the total data (37 movies) and 40% of the total data (27 movies) respectively. The model was first trained on the larger dataset and tested on the smaller. The process was repeated by reversing the roles of the training and testing data sets, essentially serving as a 2-fold crossvalidation.

Training the model:
The model was trained using a focal loss with alpha = 9 and gamma = 2.5. These values were used after observing that the complete Movie Scenes Dataset has a 9:1 ratio of negative to positive examples and the same value for gamma was reported in https://arxiv.org/abs/1708.02002 

<p align = "left">
  <img src = https://github.com/Yashgh7076/Eluvio-ML-Scene-Segmentation/blob/main/images/Fold.png width = "400" height = "300"/>
</p>

<p align = "right">
  <img src = https://github.com/Yashgh7076/Eluvio-ML-Scene-Segmentation/blob/main/images/Fold_2.png width = "400" height = "300"/>
</p>

CHALLENGES FACED:
1) The major limitation of this approach is the use of exhaustive hardware. A laptop having 16 GB of RAM can process a dataset when windows = 5, but typically encounters a MemoryError when a larger window size is chosen. This was remedied by breaking the data down into 5 parts manually and extracting features. The link to the created datasets is provided here: https://drive.google.com/drive/folders/10U2EFCuH1fP5Wc0cf7Abn9c29ppJavEx?usp=sharing

2) Due to the process being resource exhaustive, other data engineering approaches or feature transformations cannot be flexibly carried out.

