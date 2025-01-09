import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_data

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = load_data()

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the images to 1D
    layers.Dense(100, activation='relu'),  # Hidden layer with 100 units
    layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=70)

# Save the trained model
model.save('handwritten_digit_model.h5')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
