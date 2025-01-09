import tensorflow as tf
from utils import load_data

# Load the trained model
model = tf.keras.models.load_model('handwritten_digit_model.h5')

# Load the MNIST test dataset
(x_train, y_train), (x_test, y_test) = load_data()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# You can add more evaluation metrics like confusion matrix or classification report here.
