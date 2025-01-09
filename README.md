# Handwritten-Digit-Recognition  
A neural network-based handwritten digit recognition system implemented in Python using TensorFlow.  

## Overview  
This project demonstrates the implementation of a simple neural network to classify handwritten digits from the **MNIST dataset**. The neural network is designed with one hidden layer, leveraging TensorFlow for building and training the model.  

## Features  
- **Model Architecture:**  
  - 1 hidden layer with 100 units.  
  - Activation functions: ReLU and softmax.  
- **Technologies Used:**  
  - Python  
  - TensorFlow  
- **Key Techniques:**  
  - Data preprocessing  
  - Feedforward and backpropagation  
  - Regularization (lambda = 0.1)  
  - Optimizer with 70 iterations  

## Results  
The model achieved an accuracy of **92%** on the test dataset after training for 70 iterations.  

## Project Structure  
- `train.py`: Script for training the neural network model on the MNIST dataset.  
- `evaluate.py`: Script for evaluating the trained model on test data.  
- `utils.py`: Contains utility functions for data preprocessing and model evaluation.  
- `README.md`: Documentation of the project.  

## Setup and Installation  
To run this project on your local machine, follow these steps:  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Handwritten-Digit-Recognition.git
   cd Handwritten-Digit-Recognition
2.Install dependencies:
Make sure you have Python installed. Then, install the required packages:
```bash
pip install tensorflow numpy matplotlib
```
3. Running the Project
Train the Model:
Run the train.py script to train the model on the MNIST dataset:
```bash
python train.py
 ```
4. Evaluate the Model:
Once training is complete, evaluate the model's performance using the evaluate.py script:

```bash

python evaluate.py
```

## Example Output
Training Accuracy: 92%
Loss Curve: 
Confusion Matrix:

## Future Improvements
Implement additional hidden layers for better accuracy.
Use dropout for further regularization.
Experiment with advanced optimizers like Adam or RMSProp.

## Contributing
Contributions are welcome! If you'd like to contribute, feel free to fork the repository and submit a pull request.

## Contact
For questions or feedback, please reach out to:

GitHub: skiing22

Email: ramolashubham02@gmail.com

## Results
Achieved an accuracy of 92% on the test set after 70 iterations.

## License
This project is licensed under the MIT License.
