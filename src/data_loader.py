import numpy as np

def load_data(data_dir="./data"):
    """
    Load MNIST data from numpy files.
    
    Args:
        data_dir (str): Directory containing the data files
        
    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    train_images = np.load(f"{data_dir}/train_images.npy")
    train_labels = np.load(f"{data_dir}/train_labels.npy")
    test_images = np.load(f"{data_dir}/test_images.npy")
    test_labels = np.load(f"{data_dir}/test_labels.npy")
    
    return train_images, train_labels, test_images, test_labels

def get_batch(images, labels, batch_size):
    """
    Generate random mini-batches from the training data.
    
    Args:
        images (numpy.ndarray): Training images
        labels (numpy.ndarray): Training labels
        batch_size (int): Size of each mini-batch
        
    Returns:
        tuple: (batch_images, batch_labels)
    """
    indices = np.random.permutation(images.shape[0])[:batch_size]
    return images[indices], labels[indices]