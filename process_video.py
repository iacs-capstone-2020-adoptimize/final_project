import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from video_utils import CatVideo
from yolo_training.Detector import detect_raw_image
import csv
import cpbd

lin_params = np.load("model_params/lin_params.npy")
log_params = np.load("model_params/log_params.npy")
log_params_2 = np.load("model_params/log_params_2.npy")


# Citation: This code uses the following tutorial as a base and builds on top of it.
# https://blogs.oracle.com/meena/cat-face-detection-using-opencv


def run_cascade_cat_ext(gray_img):
    cat_ext_cascade = cv2.CascadeClassifier(
        "initial_exploration/haarcascade_frontalcatface_extended.xml"
    )
    SF = 1.05
    N = 6
    return cat_ext_cascade.detectMultiScale(gray_img, scaleFactor=SF,
                                            minNeighbors=N)


def sharpness_score(gray_img):
    # return cv2.Laplacian(gray_img, ddepth=3, ksize=3).var()
    return cpbd.compute(gray_img)


def get_head_distance(head_pixels, frame_shape):
    """Assuming head_pixels = (ximn, ymin, xmax, ymax)
     and frame_shape=(rows, columns)"""
    head_center = ((head_pixels[0] + head_pixels[2]) / 2 / frame_shape[1],
                   (head_pixels[1] + head_pixels[3]) / 2 / frame_shape[0])
    center = (0.5, 0.5)
    return np.linalg.norm(np.subtract(head_center, center))



# def get_features(filename, sample_rate=10, output_frames=False):
#     video = CatVideo(filename)
#     # Images for each frame where head detected; only returned if output_frames
#     head_frames = list()
#     frame_count = 0  # Total number of frames
#     cat_detected_frames = list()  # Frames in which cat head detected
#     cat_head_location = list()
#     # Of cat heads detected, variance of Laplacian inside box with head
#     sharpness = list()
#     # Of cat heads detected, ratio of cat head size to frame size
#     head_ratio = list()
#
#     for frame in video.iter_all_frames():
#         if frame_count % sample_rate == 0:
#             # Be careful! cv2 has default BGR; CatVideo has default RGB
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#             cats_found = run_cascade_cat_ext(gray_frame)
#             if len(cats_found) >= 1:
#                 if output_frames:
#                     head_frames.append(frame)
#                 cat_detected_frames.append(frame_count)
#                 cat_head_location.append(cats_found[0])
#                 x, y, w, h = cats_found[0]  # Location of first cat head
#                 head_ratio.append(w * h / (frame.shape[0] * frame.shape[1]))
#                 sharpness.append(sharpness_score(gray_frame[y:y + h, x:x + w]))
#         frame_count += 1
#     output = {
#         "frame_count": frame_count, "cat_detected_frames": cat_detected_frames,
#         "cat_head_location": cat_head_location, "sharpness": sharpness,
#         "head_ratio": head_ratio
#     }
#     if output_frames:
#         output["head_frames"] = head_frames
#     return output
#
#
# def score_video(processed_video):
#     """
#     Returns the frame number of the best picture as determined by our
#     developed model.
#     """
#     if len(processed_video["cat_detected_frames"]) == 0:
#         raise ValueError("No cats found")
#     # Of the frames where a cat head was detected
#     # get the one that scored highest
#     return processed_video["cat_detected_frames"][np.argmax(
#         rankdata(-np.abs(np.subtract(processed_video["head_ratio"], 0.05)))
#         + rankdata(processed_video["sharpness"])
#     )]
#
#
# def score_video_baseline(processed_video):
#     """
#     Returns the frame number for baseline model.
#     """
#     if len(processed_video["cat_detected_frames"]) == 0:
#         raise ValueError("No cats found")
#     return processed_video["cat_detected_frames"][
#         np.random.randint(len(processed_video["cat_detected_frames"]))
#     ]


def get_features_video(filename, sample_rate=10):
    video = CatVideo(filename)
    cat_frames = list()
    for i, frame in enumerate(video.iter_all_frames()):
        if i % sample_rate == 0:
            features = get_features_frame(frame)
            if np.any(features != 0):
                cat_frames.append(np.insert(features, 0, i))
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


def detect_cat(img):
    cat_cascade = cv2.CascadeClassifier('initial_exploration/haarcascade_frontalcatface.xml')
    cat_ext_cascade = cv2.CascadeClassifier('initial_exploration/haarcascade_frontalcatface_extended.xml')
    eye_cascade = cv2.CascadeClassifier('initial_exploration/haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # This function returns tuple rectangle starting coordinates x,y, width, height
    scale_factor = 1.05
    neighbors = 6
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                        minNeighbors=neighbors)
    cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                                minNeighbors=neighbors)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                        minNeighbors=neighbors)
    return (img, cats, cats_ext, eyes)


def create_data_for_model(file_name="labeled_results.csv"):
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
        cat = CatVideo("./videos/" + filename)
        frame = cat.get_frame_time(t)
        x_values.append(get_features_frame(frame))

    return np.array(x_values), np.array(y_values)


if __name__ == "__main__":
    # test_video = CatVideo("videos/cat1.mp4")
    # processed_video = get_features(test_video.file)
    # baseline_image = score_video_baseline(processed_video)
    # chosen_image = score_video(processed_video)
    # fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    # ax[0].imshow(test_video.get_frame_num(baseline_image))
    # ax[0].set_title("Baseline")
    # ax[0].set_axis_off()
    # ax[1].imshow(test_video.get_frame_num(chosen_image))
    # ax[1].set_title("Chosen")
    # ax[1].set_axis_off()
    # for i in range(1, 80):
    #     np.save("video_features/cat{}".format(i), get_features_video("videos/cat{}.mp4".format(i)))
    baseline_model_output = []
    lin_model_output = []
    log_model_output = []
    log_model_2_output = []
    for i in range(31, 80):
        features = np.load("video_features/cat{}.npy".format(i))
        if len(features) >= 1:
            baseline_model_output.append(["cat{}.mp4".format(i),
                                          score_video_baseline(features)])
            lin_model_output.append(["cat{}.mp4".format(i),
                                     score_video_lin(features)])
            log_model_output.append(["cat{}.mp4".format(i),
                                     score_video_log(features)])
            log_model_2_output.append(["cat{}.mp4".format(i),
                                     score_video_log_2(features)])
    import csv
    with open("model_results/baseline_model.csv", "a") as out_file:
        csv.writer(out_file).writerows(baseline_model_output)
    with open("model_results/lin_model.csv", "a") as out_file:
        csv.writer(out_file).writerows(lin_model_output)
    with open("model_results/log_model.csv", "a") as out_file:
        csv.writer(out_file).writerows(log_model_output)
    with open("model_results/log_model_2.csv", "a") as out_file:
        csv.writer(out_file).writerows(log_model_2_output)
