##Preparing the 1st Model: Single layer Perceptron
##This model is the most basic sequential model with 0 hidden layers in it.

##Adding the model layer
##We will be building the simplest model defined in the Sequential class as a linear stack of Layers

model.add(Dense(10, input_shape=(784,))
# This is same as:
model.add(Dense 10 , input_dim 784 ,))
# And to the following:
model.add(Dense 10 ,batch_input_ None 784 )))

#NOTE:

#Here the model will take input array of shape (*, 784) and outputs array of shape (*, 10).
#Dense layer is a fully connected layer and the most common type of layer used on multi layer perceptron models
