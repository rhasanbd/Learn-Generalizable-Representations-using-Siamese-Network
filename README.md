# Learn Generalizable Representations using Siamese Network

In these notebooks, our goal is to learn generalizable representations (features) from a dataset for solving classification problems. Specifically, we intend to create representations that combine two properties: (i) class discriminativeness and (ii) invariance. The latter property is broad, hence we will only focus on the rotation-invariance.

## Approach
For learning generalizable representations, we use the Siamese network architecture, which consists of two types of layers: shared layers and task-specific layers. A Siamese network or Twin network is a neural network architecture that contains two or more identical subnetworks used to generate representations (feature vectors or embeddings) for multiple input samples and compare them. Specifically, the subnetworks learn embeddings of their respective inputs via weight sharing.


We train the Siamese network to solve two classification tasks simultaneously: the main task (predicting the class labels) and an auxiliary task (e.g., predicticting the rotation angles). For solving the auxiliary task, we augment the source dataset (e.g., rotating each image either by 90 degree or 180 degree). During training, the shared layers align the representations of the two datasets.


The Siamese network architecture used in this notebook consists of two identical subnetworks (that contains the shared layers). Inputs of the original task and the auxiliary task are passed through this shared network. At the end of the shared network, a pair of task specific Dense layers is appended for these two tasks, which is followed by a classification layer.
- Original Task (supervised): predict the original label
- Auxiliary task (self-supervised): predict the rotation (angle) of the original image. A new dataset is created that contains rotated versions (by 90 degree or 180 degree) of the original images. Depending on the rotation angle, a new label is attached with each rotated image (label 1 for rotation degree 90, and 2 for rotation degree 180).

The Siamese network is trained to solve these two classification problems using the cross-entropy loss function. At the same time, it tries to minimize the Euclidean distance between the embeddings (at the task-specific layer of the network) of the pairs of images. For distance minimization, the contrastive loss function is used.

After training, the shared layers will create alignment in the representations. Also, the task-specific layers will be forced to share high-level representations, as is desirable for inducing alignment. We expect these representations to be generalizable by combining (i) class discriminativeness and (ii) rotation-invariance. 

Finally, we will use the high-level representations (task-specific layer) for the main task to predict the class labels of the augmented data (i.e., for each rotated image, the network will predict its class label).


## Experiments

Using the Siamese network, we perform three experiments.
- Experiment 1: the Siamese Network Minimizes the Distance of the Pairs of Embeddings via Contrastive Learning
- Experiment 2: the Siamese Network Does not Minimize the Distance of the Pairs of Embeddings
- Experiment 3: a Vanilla Neural Network is Trained on the Original Data 


## Notebooks
- Notebook 1: MNIST handwritten digits dataset
- Notebook 2: MNIST fashion dataset




