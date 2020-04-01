import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from utils import CatVideo

# Citation: This code uses the following tutorial as a base and builds on top of it.
# https://blogs.oracle.com/meena/cat-face-detection-using-opencv


def run_cascade_cat_ext(gray_img):
    cat_ext_cascade = cv2.CascadeClassifier('initial_exploration/haarcascade_frontalcatface_extended.xml')
    SF = 1.05
    N = 6
    return cat_ext_cascade.detectMultiScale(gray_img, scaleFactor=SF, minNeighbors=N)


def sharpness_score(gray_img):
    return cv2.Laplacian(gray_img, ddepth=3, ksize=3).var()


def get_features(filename, sample_rate=10, output_frames=False):
    video = CatVideo(filename)
    head_frames = list()  # Images for each frame where head detected; only returned if output_frames
    frame_count = 0  # Total number of frames
    cat_detected_frames = list()  # Frames in which cat head detected
    cat_head_location = list()
    sharpness = list()  # Of cat heads detected, variance of Laplacian inside box with head
    head_ratio = list()  # Of cat heads detected, ratio of cat head size to frame size

    for frame in video.iter_all_frames():
        if frame_count % sample_rate == 0:
            # Be careful! cv2 has default BGR; CatVideo has default RGB
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cats_found = run_cascade_cat_ext(gray_frame)
            if len(cats_found) >= 1:
                if output_frames:
                    head_frames.append(frame)
                cat_detected_frames.append(frame_count)
                cat_head_location.append(cats_found[0])
                x, y, w, h = cats_found[0]  # Location of cat, assuming only one cat
                head_ratio.append(w * h / (frame.shape[0] * frame.shape[1]))
                sharpness.append(sharpness_score(gray_frame[y:y + h, x:x + w]))
        frame_count += 1
    output = {"frame_count": frame_count, "cat_detected_frames": cat_detected_frames,
              "cat_head_location": cat_head_location, "sharpness": sharpness, "head_ratio": head_ratio}
    if output_frames:
        output["head_frames"] = head_frames
    return output


def score_video(processed_video):
    """
    Returns the frame number of the best picture as determined by our developed model.
    """
    if len(processed_video["cat_detected_frames"]) == 0:
        raise ValueError("No cats found")
    return processed_video["cat_detected_frames"][ # Of the frames where a cat head was detected
        np.argmax(rankdata(-np.abs(np.subtract(processed_video["head_ratio"], 0.05))  # get the one that scored highest
                           + rankdata(processed_video["sharpness"])))
    ]


def score_video_baseline(processed_video):
    """
    Returns the frame number for baseline model.
    """
    if len(processed_video["cat_detected_frames"]) == 0:
        raise ValueError("No cats found")
    return processed_video["cat_detected_frames"][np.random.randint(len(processed_video["cat_detected_frames"]))]


if __name__ == "__main__":
    test_video = CatVideo("videos/cat1.mp4")
    processed_video = get_features(test_video.file)
    baseline_image = score_video_baseline(processed_video)
    chosen_image = score_video(processed_video)
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax[0].imshow(test_video.get_frame_num(baseline_image))
    ax[0].set_title("Baseline")
    ax[0].set_axis_off()
    ax[1].imshow(test_video.get_frame_num(chosen_image))
    ax[1].set_title("Chosen")
    ax[1].set_axis_off()

    plt.show()
