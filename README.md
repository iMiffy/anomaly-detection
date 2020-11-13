Anomaly Detection
==============

This repository herein contains implementations of the following:
1. Data pre-processing using <code>imgaug</code> and <code>cv2</code>
2. Image histogram of a given input image
3. Class activation map (CAM) of a given input image using a pre-trained network (VGG16)
4. Performing feature extraction using a pre-trained VGG16 and writing into a HDF5 file, then trains a logistic regression model to fit the data
5. Performing feature extraction using a pre-trained VGG16 and writing into a HDF5 file, then trains a one class SVM model to fit the data
6. Training a convolutional autoencoder, output the reconstruction loss at test time

# Requirements
- cv2
- Pillow
- imgaug
- tensorflow >= 2.0
- sklearn
- h5py
- pickle
- progressbar


# Examples
To perform data-processing:

<code>$ python data_preprocessing.py --path images/cats_00008.jpg --copies 2</code>

![image screenshot](/images/cats_00010.jpg?raw=true)
![CAM screenshot](/output/cats_00010_augment_1.jpg?raw=true)
![CAM screenshot](/output/cats_00010_LAB.jpg?raw=true)

To get the image histogram of an image:

<code>$ python histogram_visualization.py --input images/cats_00004.jpg --output output/cats_00004_hist.jpg</code>

![image screenshot](/images/cats_00004.jpg?raw=true)
![histogram screenshot](/output/cats_00004_hist.jpg?raw=true)

To get the class activation map of an image:

<code>$ python class_activation_map.py --input images/cats_00008.jpg --output output/cats_00008_CAM.jpg</code>

![image screenshot](/images/cats_00008.jpg?raw=true)
![CAM screenshot](/output/cats_00008_CAM.jpg?raw=true)

To perform training via logistic regression on a dataset:

<code>$ python extract_features.py --dataset datasets/cats/images --output datasets/cats/hdf5/features.hdf5</code>
<code>$ python logistic_regression.py --db /datasets/cats/hdf5/features.hdf5 --model cat_LR.cpickle</code>

To perform training via one class SVM on a dataset:

<code>$ python extract_features.py --dataset datasets/cats/images --output datasets/cats/hdf5/features.hdf5</code>
<code>$ python oneclass_svm.py --db /datasets/cats/hdf5/features.hdf5 --model cat_OCSVM.cpickle</code>

To perform training and testing via convolutional autoencoder on a dataset:

<code>$ python convolutional_ae.py --image-path datasets/cats/images --model-path CAE.h5 --mode train --epoch 10 --batch-size 16 --valsplit-ratio 0.2</code>
<code>$ python convolutional_ae.py --image-path datasets/cats/test_images --model-path CAE.h5 --mode test</code>





