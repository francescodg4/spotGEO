import json
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage import util
from skimage import io
from skimage import exposure
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage import registration
from skimage import transform
from skimage import color
from skimage import feature
from sklearn import metrics
from skimage import draw
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import multiprocessing

intermediate = {

}

# Drawing functions use (row, cols) coords, transformation functions use (x, y) coords

PATH='./spotGEO/'
MODEL_PATH="mainfilter-2020_06_18.h5"

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


def get_sequence(seqid, train=False):
    # path = os.path.join(PATH, "train/")
    
    if train:
        # path = os.path.join(PATH, f"train/{seqid}/{i+1}.png")
        path = os.path.join(PATH, "train/")
    else:
        path = os.path.join(PATH, "test/")
        # path = os.path.join(PATH, f"test/{seqid}/{i+1}.png")
    
    frames = [util.img_as_float(io.imread(os.path.join(path, f'{seqid}/{i+1}.png'), as_gray=True)) for i in range(0, 5)]
    frames = np.array(frames)    
    return frames


def get_targets(seqid):
    '''Returns array of n_objects x n_frames x coords'''
    return np.array([dataset[seqid][frame] for frame in dataset[seqid]]).swapaxes(0, 1)


def draw_circle(ax, r, c, radius, color):
    circle = plt.Circle((c, r), radius, color=color, fill=False)
    ax.add_artist(circle)

    
def draw_target(ax, r, c):
    draw_circle(ax, r, c, radius=10, color='r')
    
    
def draw_prediction(ax, r, c):
    draw_circle(ax, r, c, radius=10, color='g')
    
    
def extract_region(arr, r0, c0, radius):
    """ Returns the values within a radius of the given x, y coordinates """
    return arr[(r0 - radius) : (r0 + radius + 1), (c0 - radius) : (c0 + radius + 1)]


def filter_valid_regions(regions):
    return [region for region in regions if region.shape[0] == region.shape[1]]

def filter_valid(regions, coords):
    isel = [region.shape[0] == region.shape[1] and region.shape[0] != 0 for region in regions]

    regions = [regions[i] for i, cond in enumerate(isel) if cond]
    coords = coords[isel]    
    
    return regions, coords

def register(src, dst):
    shifts, _, _ = registration.phase_cross_correlation(src, dst, upsample_factor=100)
    # phase_cross_correlation returns y, x numpy coords, we flip it to x, y
    return transform.SimilarityTransform(translation=np.flip(shifts))


def evaluate_transform_matrix(frames):
    T1 = register(frames[1], frames[0])
    T2 = register(frames[2], frames[1])
    T3 = register(frames[3], frames[2])
    T4 = register(frames[4], frames[3])

    # Sometimes the transformation matrix may return outlier transformations
    # here we assume the same transformation matrix repeated for all the frames
    Q = np.array([T.params.ravel() for T in [T1, T2, T3, T4]])

    T = np.median(Q, axis=0).reshape(3,3)
    return T


def evaluate_transforms_to_frame2(frames):
    T = evaluate_transform_matrix(frames)
    return [T @ T, T, np.eye(3), np.linalg.inv(T), np.linalg.inv(T @ T)]


def transform_sequence_to_frame2(frames):
    Ts = evaluate_transforms_to_frame2(frames)
    return np.array([warp(frame, Ti) for Ti, frame in zip(Ts, frames)])


def warp(frame, T):
    return transform.warp(frame, np.linalg.inv(T))


def get_proposal(registered, frameid):
    Ibase = np.max(registered[np.arange(len(registered)) != frameid], axis=0)

    mask = (registered[frameid] > 0)
    Idiff = np.maximum((registered[frameid]*mask) - (Ibase), 0)
    
    # Find local maxima as x, y coords
    Imaxima = morphology.h_maxima(Idiff, 0.02, selem=morphology.diamond(5))    
    r, c = np.where(Imaxima > 0)
    coords = np.c_[c, r]

    return coords


def find_targets(arr, targets):
    ''' Find regions indices corresponding to target '''
    d = metrics.pairwise_distances(arr, targets)
    return np.argmin(d, axis=0)


def coords_from_binary(arr):
    r, c = np.where(arr > 0)
    return np.c_[c, r]


def remove_stars(registered, coords):
    ''' Remove stars and recompute coordinates '''
    Ifull = np.max(registered, axis=0)

    Imaxima = np.zeros(Ifull.shape, dtype='bool')
    Imaxima[coords[:, 1], coords[:, 0]] = 1.0

    Imask = Imaxima * (Ifull < 0.6)
    
    coords = coords_from_binary(Imask > 0)
    
    return coords


def support_estimate(Ds, nns, sup_i, est_i, tol=0.05, radius=100):
    supports = []
    
    sup_i0, sup_i1 = sup_i # Support indexes
    
    neigh_dist, _ = nns[sup_i1].radius_neighbors(Ds[sup_i0], radius=np.inf)
    idxs_sorted = [np.argsort(neigh_d) for neigh_d in neigh_dist]
    
    for j_sup0, p_sup0 in enumerate(Ds[sup_i0]):        
        for j_sup1 in idxs_sorted[j_sup0]:    
            support = -1*np.ones(5, dtype='int')
            
            support[sup_i0] = j_sup0
            support[sup_i1] = j_sup1            

            # Use support pair to fit a linear model
            p_sup0 = Ds[sup_i0][j_sup0]
            p_sup1 = Ds[sup_i1][j_sup1]
            
            if np.linalg.norm(p_sup0 - p_sup1) > radius:
                 continue
                
            v = np.array([p_sup0, p_sup1])
            t = np.array([[sup_i0], [sup_i1]])

            lr = LinearRegression().fit(t, v)

            # Use this pair to estimate subsequent frames
            t_est = np.array(est_i).reshape(-1, 1)
            est = lr.predict(t_est)
            
            # Check if estimates match points in subsequent frames
            for j, e in enumerate(est):
                dist, idxs = nns[est_i[j]].radius_neighbors([e], radius=tol)
                
                if dist[0].size > 0:                    
                    closest_idx = idxs[0][np.argmin(dist[0])]
                    # Add closest match to support or negative index if no match is found
                    support[est_i[j]] = closest_idx
                else:
                    support[est_i[j]] = -1 

            # Add to support only if at least 3 matches are found
            count = np.sum(support >= 0)
            
            if np.sum(support >= 0) >= 3:               
                supports.append( support )
        
    return supports


def remove_subsequences(supports):

    output = supports.copy()

    for i, support in enumerate(supports):
        a = support
        a[a == -1]
        query = a[a != -1]

        Q = supports[:, (a != -1)]

        row = np.max(supports[np.all(query == Q, axis=1)], axis=0)
        
        output[i] = row
    
    return np.unique(output, axis=0)


def find_support(Ds, radius=100, tol=5):    
    nns = [NearestNeighbors(metric='euclidean').fit(Ds[i]) for i in range(0, 5)]
        
    # Forward pass    
    supports = support_estimate(Ds, nns, [0, 1], [2, 3, 4], radius=radius, tol=tol)
    supports += support_estimate(Ds, nns, [1, 2], [3, 4], radius=radius, tol=tol)
    supports += support_estimate(Ds, nns, [2, 3], [4], radius=radius, tol=tol)
    # Backward pass
    supports += support_estimate(Ds, nns, [4, 3], [2, 1, 0], radius=radius, tol=tol)
    supports += support_estimate(Ds, nns, [3, 2], [1, 0], radius=radius, tol=tol)
    supports += support_estimate(Ds, nns, [2, 1], [0], radius=radius, tol=tol)

    supports = np.array(supports)

    supports = remove_subsequences(supports) if len(supports) > 0 else supports

    return supports


def interpolate(trajectory):
    ''' Interpolate missing values in trajectory '''
    trajectory = np.array(trajectory)

    lr = LinearRegression().fit(trajectory[:, 0].reshape(-1, 1), trajectory[:, [1, 2]])
    
    coords = []

    for frameid in range(0, 5):
        c = trajectory[trajectory[:, 0] == frameid][:, [1, 2]]
        
        if len(c) > 0:
            coords.append(c[0])
        else:
            pred = lr.predict([[frameid]])
            coords.append(pred[0])            
        
    return np.array(coords)


# Start of main program
FIND_SUPPORT_RADIUS=40
FIND_SUPPORT_TOLERANCE=3
MEMORY_LIMIT = 500
N_PROCS = 7

DO_TRAIN = False

if DO_TRAIN:
    MAX_SEQID = 1280 # train
else:
    MAX_SEQID = 5120 # train


def run(seqid):
    print(seqid, '/', MAX_SEQID)    
    return process(seqid)


def main():
    
    pool = multiprocessing.Pool(processes=N_PROCS)

    annos = pool.map(run, range(1, MAX_SEQID + 1))
    
    pool.close()

    # Flatten list
    annotations = []
    
    for anno in annos:
        annotations += anno
    
    date = datetime.datetime.now().strftime("%F-%H%M%S")
    
    with open(f'output-{date}.json', 'w') as fp:
        json.dump(annotations, fp)

    return
    

def process(seqid):
    frames = get_sequence(seqid, train=DO_TRAIN)
	
    intermediate["frames"] = frames

    # Filter with CNN
    Iouts = []


    from keras.models import load_model
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
        except RuntimeError as e:
            print(e)

    model = load_model(MODEL_PATH)
            
    
    for frameid in range(0, 5):
        frame = frames[frameid]
        x_in = frame[..., None]
        Iout = model.predict(np.array([x_in]))[0, :, :, 0]    
        Iout = filters.median(Iout)
        Iouts.append(Iout)

    intermediate["iouts"] = Iouts
    
    # Register to reference frame
    Ts = evaluate_transforms_to_frame2(Iouts)
    intermediate["Ts"] = Ts

    registered_frames = [transform.warp(frame, np.linalg.inv(Ti)) for frame, Ti in zip(frames, Ts)]
    registered = transform_sequence_to_frame2(Iouts)

    intermediate["registered"] = registered

    # Compute background subtraction
    Idiffs = []

    for frameid in range(0, 5):
        Ibase = np.max([registered[i] for i in range(0, 5) if i != frameid], axis=0)
        Idiff = filters.gaussian(registered[frameid]) - Ibase
        Idiffs.append(Idiff)

    # Obtain proposal coordinates

    f_coords = []

    for frameid in range(0, 5):
        Idiff = Idiffs[frameid]
        Iproposal = morphology.h_maxima(Idiff, 0.3)
        fi_coords = coords_from_binary(Iproposal)
        f_coords.append( fi_coords )
    
    f_coords = np.array(f_coords)

    intermediate["f_coords"] = f_coords

    # Find support lines in coordinates
    supports = find_support(f_coords, radius=FIND_SUPPORT_RADIUS, tol=FIND_SUPPORT_TOLERANCE)

    trajectories = [[np.append(i, f_coords[i][support[i]]) for i in range(0, 5) if support[i] != -1] for support in supports]

    # Interpolate trajectories
    interpolated = [interpolate(trajectory) for trajectory in trajectories]

    intermediate["interpolated"] = interpolated

    # Convert to frame-first
    object_coords = []

    for frameid in range(0, 5):
        if len(interpolated) > 0:
            coords = np.stack([c[frameid] for c in interpolated])
            object_coords.append(coords)

    # Transform to original reference frame
    for frameid, coords in enumerate(object_coords):
        object_coords[frameid] = transform.matrix_transform(coords, np.linalg.inv(Ts[frameid]))

    # Check objects coords are within limits
    for frameid, coords in enumerate(object_coords):
        coords[:, 0] = np.clip(coords[:, 0], -0.5, 639.5)
        coords[:, 1] = np.clip(coords[:, 1], -0.5, 479.5)

    # Only first 30 objects
    object_coords = [o.tolist() for o in object_coords[:30]]

    # Output predictions
    predictions = []

    for frameid in range(0, 5):
        coords = object_coords[frameid] if len(object_coords) == 5 else []
        prediction = {'sequence_id': seqid, 'frame': frameid + 1, 'num_objects': len(coords), 'object_coords': coords}
        predictions.append(prediction)

    return predictions


if __name__ == '__main__':
    # global dataset

    # dataset = read_annotation_file(os.path.join(PATH, 'train_anno.json'))
    
    # print('Dataset contains', len(dataset), 'elements')

    main()

    
