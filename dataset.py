#installing the latest version of tensorflow
!pip install tensorflow

#verify the installation

import tensorflow as tf
from tensorflow import keras 

#Check tf.keras version
print(tf.keras.__version__)

# Loading MNIST dataset
mnist = keras.datasets.mnist 

#Splitting into train and test
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# Data Exploration
print(X_train.shape)
print(X_test.shape)

# X_train is 60000 rows of 28x28 values; we reshape it to # 60000 x 784. 
RESHAPED = 784 # 28x28 = 784 neurons
X_train = X_train.reshape(60000, RESHAPED) 
X_test = X_test.reshape(10000, RESHAPED) 

# Data is converted into float32 to use 32-bit precision # when training a neural network 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32') 

# Normalizing the input to be within the range [0,1]
X_train /= 255
#intensity of each pixel is divided by 255, the maximum intensity value
X_test /= 255
print(X_train.shape[0], 'train samples') 
print(X_test.shape[0], 'test samples') 

# One-hot representation of the labels.
Y_train = tf.keras.utils.to_categorical(Y_train, 10) 
Y_test = tf.keras.utils.to_categorical(Y_test, 10)
