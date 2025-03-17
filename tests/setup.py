import numpy as np
import struct
import os


def load_mnist_images(file_name):
    with open(file_name, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            num_images, rows * cols
        )

    return images / 255.0


def load_mnist_labels(file_name):
    with open(file_name, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    labels_one_hot = np.eye(10)[labels]

    return labels_one_hot


if not os.path.exists("data"):
    os.makedirs("data")

train_images_path = "train-images-idx3-ubyte"
train_labels_path = "train-labels-idx1-ubyte"
test_images_path = "t10k-images-idx3-ubyte"
test_labels_path = "t10k-labels-idx1-ubyte"

train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

np.save("data/train_images.npy", train_images)
np.save("data/train_labels.npy", train_labels)
np.save("data/test_images.npy", test_images)
np.save("data/test_labels.npy", test_labels)

print("MNIST data has been loaded and saved as .npy files in the 'data' directory.")
