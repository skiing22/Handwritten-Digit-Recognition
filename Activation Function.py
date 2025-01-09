##Activation function is defined in the dense layer of the model and is used to squeeze the value within a particular range. In simple term it is a function which is used to convert the input signal of a node to an output signal. tf.keras comes with the following predefined activation functions to choose from:
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
model_1 = Sequential()

# Now the model will take as input arrays of shape (*, 784)# and output arrays of shape (*, 10)
model_1.add(Dense(10,input_shape=(784,),name='dense_layer', activation='softmax'))

##In the above code we are importing the sequential keras model with 0 hidden layers. We have defined the output layer as 10. This is our dense layer. 10 is chosen as we have numbers from 0 to 9 to be classified in the dataset. shape. Total number of neurons in the input layer is 784. The activation function chosen in the dense layer is softmax. We will learn more about the softmax function in detail in our next blog. In simple terms, the model will have 784 input neurons to give the output between 0-9 numbers.

# Compiling the model next step is to compile the model. For compiling we need to define three parameters: optimizer, loss, and metrics.

#Syntax:

model.compile (optimizer=…, loss=…, metrics = …)

#1. Optimizer: While training a deep learning model, we need to alter the weights of each epoch and minimize the loss function. An optimizer is a function or algorithm that adjusts the neural network’s properties such as weights and learning rate. As a result, it helps to reduce total loss and enhance accuracy of your model.

#Some of the popular Gradient Descent Optimizers are:

##  SGD: Stochastic gradient descent, to reduce the computation cost of gradient
##RMSprop: Adaptive learning rate optimization method which utilizes the magnitude of recent gradients to normalize the gradients
##Adam: Adaptive Moment Estimation (Adam) leverages the power of adaptive learning rates methods to find individual learning rates for each parameter


sgd = SGD (...)
model. compile (optimizer = sgd)


##Loss: Loss functions are a measure of how well your model predicts the predicted outcome.


#mse : for mean squared error
#binary_crossentropy:for binary logarithmic loss (logloss)
#categorical_crossentropy: for multi class logarithmic loss (logloss)


model.compile(optimizer= adam '',loss='mse',metrics=['accuracy']
)
Copy code
Let’s put them together in the code:

# Compiling the model.
model_1.compile(optimizer='SGD', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
Model will be now trained on the on the training data. For this we will be defining the epochs, batchsize, and validation size

epoch: Number of times that the model will run through the training dataset
batch_size: Number of training instances to be shown to the model before a weight is updated
validation_split: Defines the fraction of data to be used for validation purpose
Syntax:

model.fit(X, y, epochs=..., batch_size =.., validation_split =..)
Copy code
Let’s put it together in the code,

# Training the model. 
training = model_0.fit(X_train, Y_train, batch_size=64, epochs=70, validation_split=0.2)
Copy code
Output:

Epoch 1/70

750/750 [==============================] – 1s 2ms/step – loss: 1.0832 – accuracy: 0.7526 – val_loss: 0.6560 – val_accuracy: 0.8587

Epoch 2/70

750/750 [==============================] – 1s 2ms/step – loss: 0.6081 – accuracy: 0.8562 – val_loss: 0.5083 – val_accuracy: 0.8778

Epoch 3/70

750/750 [==============================] – 1s 2ms/step – loss: 0.5130 – accuracy: 0.8701 – val_loss: 0.4506 – val_accuracy: 0.8865

Epoch 4/70

750/750 [==============================] – 1s 2ms/step – loss: 0.4667 – accuracy: 0.8784 – val_loss: 0.4181 – val_accuracy: 0.8929

From the above output you can see that with each epoch the loss is reduced and the val_accuracy is being improved.

Plot the change in accuracy and loss per epochs
You can plot a curve to check the variation of accuracy and loss as the number of epochs increases. For this you can use, matplotlib to plot the curve.

import matplotlib.pyplot as plt
%matplotlib inline

# list all data in training
print(training.history.keys())

# summarize training for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize traning for loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
