import numpy as np
import cv2
# import pylab
# import imageio

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from scipy.stats import rankdata

# cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
cat_ext_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Citation: This code uses the following tutorial as a base and builds on top of it.
# https://blogs.oracle.com/meena/cat-face-detection-using-opencv

SF = 1.05
N = 6


def run_cascade_cat_ext(gray_img):
    return cat_ext_cascade.detectMultiScale(gray_img, scaleFactor=SF, minNeighbors=N)

def sharpness_score(gray_img):
    return cv2.Laplacian(gray_img, ddepth=3, ksize=3).var()

def process_video(filename, sample_rate=10, output_frames=False):
    capture = cv2.VideoCapture(filename)
    head_frames = list()  # Images for each frame where head detected; only returned if output_frames
    frame_count = 0  # Total number of frames
    cat_detected_frames = list()  # Frames in which cat head detected
    cat_head_location = list()
    sharpness = list()  # Of cat heads detected, variance of Laplacian inside box with head
    ratio = list()  # Of cat heads detected, ratio of cat head size to frame size

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
                ratio.append(w * h / (frame.shape[0] * frame.shape[1]))
                sharpness.append(sharpness_score(
                    cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
                ))
        if not playing:
            break
        frame_count += 1
    capture.release()
    if output_frames:
        return frame_count, cat_detected_frames, cat_head_location, sharpness, ratio, head_frames
    return frame_count, cat_detected_frames, cat_head_location, sharpness, ratio


def score_video(sharpness_list, ratio_list):
    # Returns index
    return np.argmax(rankdata(-np.abs(np.subtract(ratio_list, 0.05)) + rankdata(sharpness_list)))

if __name__ == "__main__":
    processed_video = process_video("videos/MVI_3414.MP4", output_frames=True)
    plt.imshow(processed_video[-1][score_video(processed_video[3], processed_video[4])])
    plt.show()