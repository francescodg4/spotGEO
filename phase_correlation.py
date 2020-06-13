import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

from main import read_image, DATASET_PATH, larger

SEQUENCE_INDEX = 55

sequence = [read_image(DATASET_PATH, sequence=SEQUENCE_INDEX, frame=fr) for fr in range(1, 6)]

stack = []

for i in range(0, 5):
    if i == 2:
        d_row, d_col = (0, 0)
    else:
        shift, error, _ = phase_cross_correlation(sequence[2], sequence[i])
        d_row, d_col = np.round(shift)

    stack.append(larger(sequence[i], int(d_row), int(d_col)))

o = np.max(stack, axis=0)

_, axs = plt.subplots(2, 3)
axs = axs.ravel()

for ax, frame in zip(axs, sequence):
    ax.imshow(frame)

axs[-1].imshow(o)

plt.show()

plt.imshow(o)
plt.show()
