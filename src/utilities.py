import numpy as np

def calculate_accuracy(predictions, labels):
    """
    Calculate accuracy from predictions and true labels.

    Args:
        predictions (numpy.ndarray): Predicted digit classes (0-9)
        labels (numpy.ndarray): True labels (one-hot encoded)

    Returns:
        float: Accuracy as a percentage
    """
    true_labels = np.argmax(labels, axis=1)
    return np.mean(predictions == true_labels) * 100


def calculate_loss(output, y):
    """
    Calculate cross-entropy loss.

    Args:
        output (numpy.ndarray): Model output probabilities
        y (numpy.ndarray): True labels (one-hot encoded)

    Returns:
        float: Cross-entropy loss
    """

    output = np.clip(output, 1e-10, 1.0)
    return -np.mean(np.sum(y * np.log(output), axis=1))


def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.

    Args:
        labels (numpy.ndarray): Class labels as integers
        num_classes (int): Number of different classes

    Returns:
        numpy.ndarray: One-hot encoded labels
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def evaluate_model(model, test_images, test_labels, batch_size=100):
    """
    Evaluate the model on test data.

    Args:
        model (NeuralNetwork): Trained neural network model
        test_images (numpy.ndarray): Test images
        test_labels (numpy.ndarray): Test labels (one-hot encoded)
        batch_size (int): Batch size for evaluation

    Returns:
        tuple: (accuracy, loss)
    """
    num_samples = test_images.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_loss = 0
    all_predictions = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_images = test_images[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]

        output, _ = model.forward(batch_images)
        predictions = np.argmax(output, axis=1)
        all_predictions.extend(predictions)

        loss = calculate_loss(output, batch_labels)
        total_loss += loss * (end_idx - start_idx)

    avg_loss = total_loss / num_samples
    accuracy = calculate_accuracy(np.array(all_predictions), test_labels)

    return accuracy, avg_loss
