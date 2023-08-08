import torch
import torch.utils.data as data
# Numpy is a math library
import numpy as np
# Matplotlib is a graphing library
import matplotlib.pyplot as plt
# math is Python's math library
import math

# We'll generate this many sample datapoints
SAMPLES = 1000

# Set a "seed" value, so we get the same random numbers each time we run this
# notebook
np.random.seed(1337)

# Generate a uniformly distributed set of random numbers in the range from
# 0 to 2π, which covers a complete sine wave oscillation
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)

# Shuffle the values to guarantee they're not in order
np.random.shuffle(x_values)

# Calculate the corresponding sine values
y_values = np.sin(x_values)
y_values += 0.1 * np.random.randn(*y_values.shape)
# Plot our data. The 'b.' argument tells the library to print blue dots.
TRAIN_SPLIT =  int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

# Use np.split to chop our data into three parts.
# The second argument to np.split is an array of indices where the data will be
# split. We provide two indices, so the data will be divided into three chunks.
x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# Double check that our splits add up correctly
assert (x_train.size + x_validate.size + x_test.size) ==  SAMPLES

class SineData(data.Dataset):
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

    def __getitem__(self, index):#返回的是tensor
        x_value, y_value = self.x_values[index], self.y_values[index]
        return x_value, y_value

    def __len__(self):
        return len(self.x_values)

def get_data_loader(batch_size):
    train_data = SineData(x_train, y_train)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = SineData(x_validate, y_validate)
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data = SineData(x_test, y_test)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader