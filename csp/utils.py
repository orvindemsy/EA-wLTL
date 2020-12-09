import numpy as np


# function to count subject
def subject_counter(i):
    return 'subject{:02d}'.format(i)

# Load .npz data

def load_data(data_path, key):
    # New dict to store data
    data = dict()

    # Load the data into dict
    data[key] = np.load(data_path)

    return data