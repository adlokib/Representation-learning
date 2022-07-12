# Representation-learning
Representation Learning project with clustering

## Repository implementing representation learning
In cases where we want to be able to distinguish between images from different classes and images of the same class, many times it makes sense to go with an approach that embeds the images in some vector space in such a manner that images belonging to the same class are represented by very similar vectors, and images from different classes are represented as very different vectors.

A very popular example of this would be face recognition. Where the face of a person is represented as a vector of floating point values (usually 256 or 512 dimensional). In doing so, we train the model to generate embeddings for inputs instead of training to assign a class to each face.

This particular repository tackles the task of scene recognition, but the project is structured such that with minimal changes one could adapt it to any representation learning task. The structure of the approach here has been arrived at through a lot of iterations on what works best. The project can be broken down into various segments, which will be described below.

# Project Segments
## Dataset
## Batching
## Model Architecture
## Loss function
## Multi GPU Support
## Clustering

# Dataset
The dataset for training needs to be set up in map style. That is, the dataset folder must contain sub-folders for each class, and each sub-folder should contain all the images from that class. As with all model training, more the data, better the learning. Representation learning is a bit different from normal model training in the sense that, here, the number of classes is more importatnt that the images per class. As the task of the model is to learn intra-class similarity and inter-class dissimilarity.
A good idea before training would be to resize all the images in the dataset to the required image size to minimize data-loading overhead. But even if an image does not match the specified size, it will be resized during run time.
The `create_dict` function parses the dataset and creates two dictionaries for us to be able to create training batches. Each entry in the dictionary corresponds to one class. All classes should be mutually exclusive. The data_dict is a dictionary that contains list of path of all images for each class, whereas, length_dict is a dictionary that containg number of images for each class.

# Batching
The project is structured to leverage the advantages of PK batching. In PK batching, a batch is created by selecting 'P' classes and 'K' samples for each class. This has numerous advantages over conventional triplet generation, as in that case, the number of comparisions considered in one pass is the same as the batch-size. Whereas, in PK batching, the number of considerations are 
batch-size x (K-1) x ( (P-1) x K )
[num_examples X positive_examples_for_that_entry X negative_examples_for_that_entry]

That is to say, we make use of all possible valid combinations in a batch of images, but putting certain constraints on how to batch them.

To achieve this, we utilize `create_batch` function that provides us with the batch that we need. It takes in 'P', 'K' and 'length_dict' as input and generates a batch for us to process. Since we are not working with a conventional dataset, that can be covered linearly, it makes more sene to randomly create batches from our dataset as opposed to try to cover it linearly. Hence, the project only employs iterations, and not epochs. As it does not make sense to iterate over a dataset in this particular scenario.

We use the `parse_batch` helper function to parse the batch we have created and generate a numpy array for the batch we want to process. The function resizes the image, if it doesn't match with the specified image size.

Also, we apply a list of transformations to the images randomly that provides a bit of regularization to the model training

# Model Architecture
The basic architecture used in the project is-
A resnet backbone for extracting features, followed by multi-head attention encoders.

We define the dimensions as a parameter of the network. here I use 128 dimensional vectors. More importantly, the input image is not of shape 224x224 as it would yield us a 1x1 output. We take a bigger resolution as it gives us an opportunity to leverage attention when training.
When we run the same conv layers on images of larger input shape, its the same as applying the model on different slices of the images. Once we obtain our output through the backbone, we apply 1x1 conv layer, to reduce the number of dimensions to our required number of dimensions. After this, we add the positional embeddings to the output and unroll it. We then feed into three consecutive layers of multi-head attention encoder layers. Finally, we have a linear mapping from the encoder output to the final output dimension.
Overall, the architecture is fairly simple and straightforward.

# Loss function
The loss function used here is "Circle loss". The loss function is implemented in a vectorized way. We generate an NxN matrix of cosine similarities which we then use to separate the Positive similarities and the negative similaritives. Which are then futher processed. Since the implementation is vectorised, it runs much faster as opposed to using a for loop.
The advantage of using Circle loss as opposed to triplet loss is that, we can leverage PK batching as well as the fact that, circle loss has a difinitive convergence target Maximizing Sp and minimizing Sn separately. Whereas, in triplet loss what we are optimizing the distance between Sp and Sn and not thier individual values.

# Multi GPU Support
On training jobs of very large scope, e.g. where number of classes are 2.5K or even greater, it becomes necessary to employ multi GPU training. In the project, for multi GPU trining, we set the batch_size to Num_gpus x batch_for_each_gpu. The batch is then scattered to each GPU and trained.

# Clustering
The clustering carried out here is a way to group an unorganised list of images. This is different from normal Representation learning uses, as I am not employing a support set. The training carried out is capable enough to separate the vector spaces to an extent that, we do not need a support set and can straight forwardly cluster the images regardless of the number of classes.
The clustering algorithm is as follows-

We generate the embeddings of the images using the model we have trained.
We then use the embeddings to generate a similarity matrix.
We us a recursive function to tag all the images that have similarity value greater than a threshold. And since we are using a recursive function, we can traverse to all the images that can be accessed.
Upon exhausting all the possible connections, we increase the class index and call the same function on the next image that has not yet been visited.
We continue this until we have visited all the images.

We can then optionally, set all the classes that have less than a set number of images to class 0, so as to signify that these images dont belong to any class, or are difficult images to cluster.

The advantage of this clustering approach, as opposed to DBScan or K-means is that it is very straightforward in intuition as well as not limiting us to a set number of classes.

# References

https://arxiv.org/abs/2005.12872

https://openjournals.uwaterloo.ca/index.php/vsl/article/download/3533/4579

https://arxiv.org/abs/1706.03762

https://arxiv.org/abs/2002.10857

https://arxiv.org/abs/1703.07737
