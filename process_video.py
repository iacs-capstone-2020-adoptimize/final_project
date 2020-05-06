import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from video_utils import CatVideo
from yolo_training.Detector import detect_raw_image
import csv
import cpbd


# lin_params = np.load("data/regression_parameter_results/lin_params_v0.npy")
# log_params = np.load("data/regression_parameter_results/log_params_any_features_v0.npy")
# log_params_2 = np.load("data/regression_parameter_results/log_params_all_features_v0.npy")
#

def sharpness_score(gray_img):
    return cpbd.compute(gray_img)


def get_head_distance(head_pixels, frame_shape):
    """Assuming head_pixels = (ximn, ymin, xmax, ymax)
     and frame_shape=(rows, columns)"""
    head_center = ((head_pixels[0] + head_pixels[2]) / 2 / frame_shape[1],
                   (head_pixels[1] + head_pixels[3]) / 2 / frame_shape[0])
    center = (0.5, 0.5)
    return np.linalg.norm(np.subtract(head_center, center))


def get_features_video(filename, sample_rate=10, return_frames=False):
    video = CatVideo(filename)
    cat_frames = list()
    image_data = list()
    for i, frame in enumerate(video.iter_all_frames()):
        if i % sample_rate == 0:
            features = get_features_frame(frame)
            if np.any(features != 0):
                cat_frames.append(np.insert(features, 0, i))
            image_data.append(frame)
    if return_frames:
        return np.asarray(cat_frames), image_data
    else:
        return np.asarray(cat_frames)

def get_features_frame(frame):
    features_detected = detect_raw_image(frame)
    eyes = []
    noses = []
    ears = []
    heads = []
    for feature in features_detected:
        x1, y1, x2, y2, c, conf = feature
        if c == 0:
            eyes.append(feature)
        if c == 1:
            noses.append(feature)
        if c == 2:
            ears.append(feature)
        if c == 3:
            heads.append(feature)
    head_size, eye_ratio, ear_ratio = 0, 0, 0
    conf_eye_0, conf_eye_1, conf_nose, conf_ear_0, conf_ear_1, conf_head \
        = 0, 0, 0, 0, 0, 0
    head_distance = 1
    sharpness = 0
    if len(heads) > 0:
        # print(t)
        # print("Cat detected ^^")
        heads = np.array(heads)
        best_head = heads[np.argmax(heads[:, 5])]
        best_head, conf_head = best_head[:4], best_head[5]
        best_head = best_head.astype(int)
        head_size = ((best_head[2] - best_head[0])
                     * (best_head[3] - best_head[1])
                     / (frame.shape[0] * frame.shape[1]))
        gray_frame = cv2.cvtColor(frame[best_head[1]:best_head[3],
                                  best_head[0]:best_head[2]],
                                  cv2.COLOR_RGB2GRAY)
        sharpness = sharpness_score(gray_frame)
        head_distance = get_head_distance(best_head, frame.shape)

    if len(eyes) == 1:
        conf_eye_0 = eyes[0][5]
    elif len(eyes) >= 2:
        eyes = np.array(eyes)
        best_eyes = eyes[eyes[:, 5].argsort()][-2:][::-1]
        conf_eye_0, conf_eye_1 = best_eyes[:, 5]
        eyes_size = ((best_eyes[:, 2] - best_eyes[:, 0])
                     * (best_eyes[:, 3] - best_eyes[:, 1]))
        eye_ratio = eyes_size[0] / eyes_size[1]
        if eye_ratio > 1:
            eye_ratio = 1 / eye_ratio
    if len(noses) >= 1:
        conf_nose = np.max(np.array(noses)[:, 5])
    if len(ears) == 1:
        conf_ear_0 = ears[0][5]
    elif len(ears) >= 2:
        ears = np.array(ears)
        best_ears = ears[ears[:, 5].argsort()][-2:][::-1]
        conf_ear_0, conf_ear_1 = best_ears[:, 5]
        ears_size = ((best_ears[:, 2] - best_ears[:, 0])
                     * (best_ears[:, 3] - best_ears[:, 1]))
        ear_ratio = ears_size[0] / ears_size[1]
        if ear_ratio > 1:
            ear_ratio = 1 / ear_ratio
    return np.array([eye_ratio, head_size, ear_ratio, conf_head,
                     conf_eye_0, conf_eye_1, conf_ear_0, conf_ear_1,
                     conf_nose, sharpness, head_distance])


def score_video_baseline(features):
    return int(np.random.choice(features[:, 0]))


def score_video_log(features):
    classes = np.exp(features[:, 1:] @ log_params.T)
    classes = classes / np.sum(classes, axis=1)
    return int(features[np.argmax(classes[:, 3] + classes[:, 4]), 0])


def score_video_lin(features):
    return int(features[np.argmax(features[:, 1:] @ lin_params), 0])


def score_video_log_2(features):
    reduced_features = features[np.all(features[:, 1:] != 0, axis=1)]
    if len(reduced_features) == 0:
        reduced_features = features
    classes = np.exp(reduced_features[:, 1:] @ log_params_2.T)
    classes = classes / np.sum(classes, axis=1)
    return int(reduced_features[np.argmax(classes[:, 3] + classes[:, 4]), 0])


def create_data_for_model(file_name):
    """
    Creates the data points to put into the logistic regression model.
    """
    y_values = []
    files_seconds = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            #process all frames that have cats in them
            if int(row[3]) != 0:
                y_values.append(int(row[3]))
                files_seconds.append((row[1], float(row[2])))
    x_values = []
    for (filename, t) in files_seconds:
        cat = CatVideo("data/videos/" + filename)
        frame = cat.get_frame_time(t)
        x_values.append(get_features_frame(frame))

    return np.array(x_values), np.array(y_values)


if __name__ == "__main__":
    get_features_video_with_numpy_image("/data/videos/cat66.mp4")
