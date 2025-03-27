# MNIST Digit Classification using Convolutional Neural Network (CNN)

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## 📁 Dataset

- **MNIST**: The dataset contains 70,000 28x28 grayscale images of digits (0–9).
  - 60,000 images for training
  - 10,000 images for testing

## 🧠 Model Architecture

The model is a CNN built with Keras Sequential API and includes:

- Convolutional layers
- MaxPooling layers
- Dropout layer for regularization
- Flatten and Dense layers
- Final output layer with softmax activation

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install tensorflow matplotlib scikit-learn
```

### 2. Run the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook Implementing_Convolutional_Neural_Network.ipynb
```

### 3. Evaluate the Model

You can evaluate the model using the built-in Keras evaluation method:

```python
score = mnist_model.evaluate(test_images, test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

## 📊 Results

- The model achieves **high accuracy** on both training and test data.
- Visualization of training/validation loss and accuracy is included in the notebook.
- Sample predictions are shown with their actual labels.


## 📄 License

This project is for educational purposes.