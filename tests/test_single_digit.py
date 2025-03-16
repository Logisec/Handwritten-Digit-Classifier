import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from neural_network import NeuralNetwork

def load_model(model_dir="models"):
    """Load the trained model parameters."""
    W1 = np.load(os.path.join(model_dir, "W1.npy"))
    b1 = np.load(os.path.join(model_dir, "b1.npy"))
    W2 = np.load(os.path.join(model_dir, "W2.npy"))
    b2 = np.load(os.path.join(model_dir, "b2.npy"))

    input_size = W1.shape[0]
    hidden_size = W1.shape[1]
    output_size = W2.shape[1]

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    nn.W1 = W1
    nn.b1 = b1
    nn.W2 = W2
    nn.b2 = b2

    return nn


def preprocess_image(image_path):
    """Preprocess an image to match MNIST format."""
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img) / 255.0

    flattened = img_array.reshape(1, 784)

    return flattened, img_array


def predict_digit(model, image_path):
    """Predict the digit in the image."""

    img_data, img_array = preprocess_image(image_path)

    prediction = model.predict(img_data)[0]

    output, _ = model.forward(img_data)
    probabilities = output[0]

    return prediction, probabilities, img_array


def main():

    if len(sys.argv) < 2:
        print("Usage: python test_single_digit.py <path_to_image>")
        return

    image_path = sys.argv[1]

    print("Loading the trained model...")
    model = load_model()

    print("Predicting the digit...")
    prediction, probabilities, img_array = predict_digit(model, image_path)

    print(f"Predicted digit: {prediction}")
    print("Confidence levels:")
    for digit, prob in enumerate(probabilities):
        print(f"  Digit {digit}: {prob*100:.2f}%")

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap="gray")
    plt.title(f"Predicted: {prediction}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.bar(range(10), probabilities)
    plt.xticks(range(10))
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.title("Confidence Levels")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
