import json
import datetime
import skimage
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

DATASET_PATH = "./spotGEO/train/{0}/{1}.png"


def create_submission_file():
    """Write the submission file"""

    submission = []

    for sequence_id in range(1, 5121):
        for frame in range(1, 6):
            submission.append(
                {
                    "sequence_id": sequence_id,
                    "frame": frame,
                    "num_objects": 0,
                    "object_coords": [],
                }
            )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%m-%S")

    with open(f"sub_{timestamp}.json", "w") as outfile:
        json.dump(submission, outfile)


def larger(img: np.ndarray, d_row: int, d_col: int) -> np.ndarray:
    height, width = img.shape
    output = np.zeros((int(height * 2), int(width * 2)), dtype=np.uint8)

    dr = (height) // 2
    dc = (width) // 2

    output[
        (dr + d_row) : (dr + d_row + height), (dc + d_col) : (dc + d_col + width)
    ] = img

    return output


def read_image(path: str, sequence: int, frame: int):
    return cv2.imread(path.format(sequence, frame), cv2.IMREAD_GRAYSCALE)


def denoising(image: np.ndarray):
    gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(gX**2 + gY**2)
    magnitude = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )

    return cv2.GaussianBlur(magnitude, (5, 5), 0)


def align_stack(indexes, frames):
    stack = []

    for index, frame in zip(indexes, frames):
        if index == 3:
            d_row, d_col = (0, 0)
        else:
            shift, error, _ = phase_cross_correlation(frames[2], frame)
            d_row, d_col = np.round(shift)

        stack.append(larger(frame, d_row=int(d_row), d_col=int(d_col)))

    return stack


def compute_sequence_alignment(indexes, frames):
    """Compute the alignment d_row, d_col offsets to align the images to the central frame"""
    alignment = []

    for index, frame in zip(indexes, frames):
        if index == 3:
            d_row, d_col = (0, 0)
        else:
            shift, error, _ = phase_cross_correlation(frames[2], frame)
            d_row, d_col = np.round(shift)

        alignment.append((d_row, d_col))

    return alignment


def test_multiple_noise_reduction():
    def rolling_ball(image: np.ndarray):
        from skimage import restoration

        background = restoration.rolling_ball(image)
        return image - background

    def gaussian(image: np.ndarray):
        background = cv2.GaussianBlur(image / 255.0, (31, 31), 0)
        return image - background

    def gradient(image: np.ndarray):
        gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        return np.sqrt(gX**2 + gY**2)

    def mixed(image: np.ndarray):
        gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gX**2 + gY**2)
        return cv2.GaussianBlur(magnitude, (5, 5), 0)

    import matplotlib.pyplot as plt

    img = read_image(DATASET_PATH, sequence=1, frame=1)

    methods = [
        ("rolling_ball", rolling_ball),
        ("gaussian", gaussian),
        ("gradient", gradient),
        ("mixed", mixed),
    ]

    _, axs = plt.subplots(2, 2)
    axs = axs.ravel()

    for ax, (label, function) in zip(axs, methods):
        ax.imshow(function(img))
        ax.set_title(label)

    plt.show()


def main():
    import matplotlib.pyplot as plt

    sequence_index = 125

    sequence = [read_image(DATASET_PATH, sequence=sequence_index, frame=fr) for fr in range(1, 6)]
    alignment = compute_sequence_alignment(range(1, 6), sequence)

    aligned_frames = []

    for frame, (d_row, d_col) in zip(sequence, alignment):
        frame = np.uint8(denoising(frame))
        aligned_frame = larger(frame, int(d_row), int(d_col))
        aligned_frames.append(aligned_frame)

    merged = np.sort(aligned_frames, axis=0)

    import matplotlib.pyplot as plt

    plt.imshow(merged[-1] - merged[-2])
    plt.show()


if __name__ == "__main__":
    main()
