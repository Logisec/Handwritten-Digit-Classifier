import numpy as np
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_data, get_batch
from src.neural_network import NeuralNetwork
from src.utilities import evaluate_model


def main():
    input_size = 784
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01
    batch_size = 64
    epochs = 10

    print("Loading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_data()

    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    print("Initializing neural network...")
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    num_samples = train_images.shape[0]
    num_batches = num_samples // batch_size

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0

        indices = np.random.permutation(num_samples)
        shuffled_images = train_images[indices]
        shuffled_labels = train_labels[indices]

        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size

            X_batch = shuffled_images[batch_start:batch_end]
            y_batch = shuffled_labels[batch_start:batch_end]

            loss = nn.train_step(X_batch, y_batch)
            epoch_loss += loss

            if (batch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{num_batches}, Loss: {loss:.4f}")

        test_accuracy, test_loss = evaluate_model(nn, test_images, test_labels)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, "
            f"Avg. Loss: {epoch_loss/num_batches:.4f}, "
            f"Test Accuracy: {test_accuracy:.2f}%, "
            f"Test Loss: {test_loss:.4f}"
        )

    print("Training completed!")

    final_accuracy, final_loss = evaluate_model(nn, test_images, test_labels)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"Final Test Loss: {final_loss:.4f}")

    print("Saving model parameters...")
    os.makedirs("models", exist_ok=True)
    np.save("models/W1.npy", nn.W1)
    np.save("models/b1.npy", nn.b1)
    np.save("models/W2.npy", nn.W2)
    np.save("models/b2.npy", nn.b2)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
