import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# Citation: This code uses the following tutorial as a base and builds on top of it.
# https://blogs.oracle.com/meena/cat-face-detection-using-opencv


def run_cascade_cat_ext(gray_img):
    cat_ext_cascade = cv2.CascadeClassifier('initial_exploration/haarcascade_frontalcatface_extended.xml')
    SF = 1.05
    N = 6
    return cat_ext_cascade.detectMultiScale(gray_img, scaleFactor=SF, minNeighbors=N)


def sharpness_score(gray_img):
    return cv2.Laplacian(gray_img, ddepth=3, ksize=3).var()


def get_frame(filename, frame_number):
    capture = cv2.VideoCapture(filename)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    status, frame = capture.read()
    return frame


def extract_features(filename, sample_rate=10, output_frames=False):
    capture = cv2.VideoCapture(filename)
    head_frames = list()  # Images for each frame where head detected; only returned if output_frames
    frame_count = 0  # Total number of frames
    cat_detected_frames = list()  # Frames in which cat head detected
    cat_head_location = list()
    sharpness = list()  # Of cat heads detected, variance of Laplacian inside box with head
    head_ratio = list()  # Of cat heads detected, ratio of cat head size to frame size

    while capture.isOpened():
        playing, frame = capture.read()
        if playing and frame_count % sample_rate == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cats_found = run_cascade_cat_ext(gray_frame)
            if len(cats_found) >= 1:
                if output_frames:
                    head_frames.append(frame)
                cat_detected_frames.append(frame_count)
                cat_head_location.append(cats_found[0])
                x, y, w, h = cats_found[0]  # Location of cat, assuming only one cat
                head_ratio.append(w * h / (frame.shape[0] * frame.shape[1]))
                sharpness.append(sharpness_score(
                    cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
                ))
        if not playing:
            break
        frame_count += 1
    capture.release()
    output = {"frame_count": frame_count, "cat_detected_frames": cat_detected_frames,
              "cat_head_location": cat_head_location, "sharpness": sharpness, "head_ratio": head_ratio}
    if output_frames:
        output["head_frames"] = head_frames
    return output


def score_video(processed_video):
    return np.argmax(rankdata(-np.abs(np.subtract(processed_video["head_ratio"], 0.05))
                              + rankdata(processed_video["sharpness"])))


def score_video_baseline(processed_video):
    return np.random.randint(len(processed_video["cat_detected_frames"]))


if __name__ == "__main__":
    # processed_video = extract_features("initial_exploration/videos/MVI_3414.MP4", output_frames=True)
    # plt.imshow(processed_video["head_frames"][score_video(processed_video)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax1.imshow(get_frame("initial_exploration/videos/MVI_3414.MP4", 0))
    ax2.imshow(get_frame("initial_exploration/videos/MVI_3414.MP4", 10))
    plt.show()