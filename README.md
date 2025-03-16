# MNIST Digit Classification from Scratch

This project implements a neural network for handwritten digit classification using the MNIST dataset, built from scratch using only NumPy.

## Project Structure

```
handwritte-digit-classifier/
│
├── data/
│   ├── train_images.npy       # Training images
│   ├── train_labels.npy       # Training labels
│   ├── test_images.npy        # Test images
│   ├── test_labels.npy        # Test labels
│
├── src/
│   ├── __init__.py            # Marks this directory as a package
│   ├── data_loader.py         # Loads the MNIST data and preprocesses it
│   ├── neural_network.py      # Contains the neural network code (forward pass, backpropagation, etc.)
│   ├── utils.py               # Helper functions for evaluation, loss calculation, etc.
│
├── tests/
│   ├── setup.py               # MNIST Training & Testing data setup.
│   ├── test_single_digit.py   # Testing our own image.
|
├── main.py                    # The entry point to train the model and run the project
├── requirements.txt           # Python dependencies for the project
└── README.md                  # Project description and setup instructions
```

Due to GitHub's file size limitations, we are unable to upload the `data/` directory. To use the project, you will need to download the MNIST ubyte files from [MNIST Datasets Github.io](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/). After downloading the raw data, run the `setup.py` script located in the `/tests/` directory to convert the ubyte files into `.npy` format. This will automatically create the `/data/` directory and save the necessary `.npy` files for use in the project. (Make sure to put the ubyte files in the `/tests` directory)

## Features

- Implements a two-layer neural network with ReLU activation in the hidden layer and softmax activation in the output layer
- Includes forward and backward propagation from scratch
- Mini-batch gradient descent for optimization
- Cross-entropy loss function
- Progress tracking and model evaluation

## Setup and Installation

1. Clone this repository:
```
git clone https://github.com/logisec/handwritte-digit-classifier.git
cd "handwritte-digit-classifier"
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Make sure you have the MNIST data files in the correct format in the `data/` directory.

## Usage

To train the model and evaluate it on the test set, run:

```
python main.py
```

## Neural Network Architecture

- Input layer: 784 neurons (28x28 pixels flattened)
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 10 neurons with softmax activation (one for each digit)

## Hyperparameters

The default hyperparameters are:
- Learning rate: 0.01
- Batch size: 64
- Number of epochs: 10
- Hidden layer size: 128

You can modify these in the `main.py` file to experiment with different settings.

## Results

With the default settings, the model should achieve around 95-97% accuracy on the test set after 10 epochs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.