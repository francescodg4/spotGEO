import json
import numpy as np
import sys


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


d1 = read_annotation_file('./spotGEO/train_anno.json')
d2 = read_annotation_file(sys.argv[1])

def get_targets(dataset, seqid):
    '''Returns array of n_objects x n_frames x coords'''
    return np.array([dataset[seqid][frame] for frame in dataset[seqid]]).swapaxes(0, 1)


def show_min_max(targets1, targets2):

    isclose = lambda a, b: np.isclose(a, b, atol=0.1)
    
    for t1, t2 in zip(targets1, targets2):
        min_x, min_y = np.min(t2, axis=0)
        print( isclose(min_x, 0.5), isclose(min_y, 0.5))

    for t1, t2 in zip(targets1, targets2):
        max_x, max_y = np.max(t2, axis=0)
        print( isclose(max_x, 639.5), isclose(max_y, 479.5))

count = 0

for seqid in range(1, len(d1) + 1):
    targets1 = get_targets(d1, seqid)
    targets2 = get_targets(d2, seqid)

    ntargets1 = len(targets1)
    ntargets2 = len(targets2)

    if ntargets1 != ntargets2:
        print(seqid, 'true', ntargets1, 'predicted', ntargets2, (ntargets2 - ntargets1))
        count += 1

        if abs(ntargets1 - ntargets2) > 2:
            show_min_max(targets1, targets2)

print('wrong', count, count/(len(d1)+1))
