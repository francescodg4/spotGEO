from skimage import io
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np
from skimage import morphology
from skimage import filters
from skimage import exposure
from skimage import util


def read_annotation_file(path):
    '''This creates a dictionary where the first key is the sequence id,
    and the second key is the frame id, which contains a list of the 
    annotation points as lists of floating numbers.
    For example sequence #1 shown above contains 3 objects, which are visible in both frames #1 and #3.'''
    with open(path) as annotation_file:
        annotation_list = json.load(annotation_file)
    # Transform list of annotations into dictionary
    annotation_dict = {}
    for annotation in annotation_list:
        sequence_id = annotation['sequence_id']
        if sequence_id not in annotation_dict:
            annotation_dict[sequence_id] = {}
        annotation_dict[sequence_id][annotation['frame']] = annotation['object_coords']
    return annotation_dict


def get_targets(dataset, seqid):
    '''Returns array of n_objects x n_frames x coords'''
    return np.array([dataset[seqid][frame] for frame in dataset[seqid]]).swapaxes(0, 1)


def get_filename_and_coords(dataset, annotation, idx):
    filename = dataset['filenames'][idx]

    frameid = int(os.path.basename(filename).split('.')[0])
    seqid = int(dataset['target_names'][dataset['target'][idx]])

    targets = get_targets(annotation, seqid)

    coords = targets[:, (frameid-1)]

    return filename, coords


def process(Iin, targets, sigma=5):
    Iout = np.zeros(Iin.shape)
    for c, r in targets:
        Iout[int(r + 0.5), int(c + 0.5)] = 1
    Iout = filters.gaussian(Iout, sigma=sigma)
    Iout = exposure.rescale_intensity(Iout)
    return Iout


# DataGenerator
import numpy as np
import keras

class SpotGEODataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim, batch_size=32, shuffle=True, sigma=5):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.sigma = sigma
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))
        
        for i, (filename, coords) in enumerate(list_IDs_temp):
            Iin = util.img_as_float(io.imread(filename, as_gray=True))
            Iout = process(Iin, coords, sigma=self.sigma)
            
            X[i,] = Iin[:, :, None]
            y[i,] = Iout[:, :, None]
        
        return X, y


DATASET_PATH='./spotGEO/train'
ANNOTATION_PATH='./spotGEO/train_anno.json'

# Split samples in dataset in train set and test set
path = DATASET_PATH

dataset = datasets.load_files(path,
                              load_content=False,
                              shuffle=False)

anno = read_annotation_file(ANNOTATION_PATH)

# Remove empty samples from dataset
data, idxs = [], []

for idx in dataset['target']:
    filename, coords = get_filename_and_coords(dataset, anno, idx)
    if len(coords) > 0:
        data.append((filename, coords))
        idxs.append(idx)

data = np.array(data)

# Split train and test samples
train_idxs, test_idxs = train_test_split(
    idxs,
    test_size=0.3,
    random_state=0)

train_set = data[train_idxs]
test_set = data[test_idxs]

# Parameters
params = {
    'dim': (480, 640, 1),
    'batch_size': 32,
    'shuffle': True,
    'sigma': 5
}

training_generator = SpotGEODataGenerator(train_set, **params)

test_generator = SpotGEODataGenerator(test_set, **params)
