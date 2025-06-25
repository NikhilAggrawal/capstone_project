import numpy as np
import pandas as pd

# Load CSV
data = pd.read_csv('data/mnist_test.csv').values

# Split labels and images
labels = data[:, 0].astype(np.uint8)
images = data[:, 1:].astype(np.uint8)

# Save as binary
images.tofile('data/mnist_test_images.bin')
labels.tofile('data/mnist_test_labels.bin')

# Similar process for training data
train_data = pd.read_csv('data/mnist_train.csv').values  
train_labels = train_data[:, 0].astype(np.uint8)
train_images = train_data[:, 1:].astype(np.uint8)
# Save training data as binary
train_images.tofile('data/mnist_train_images.bin')
train_labels.tofile('data/mnist_train_labels.bin')