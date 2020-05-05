import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from video_utils import CatVideo
from yolo_training.Detector import detect_raw_image
import csv
import cpbd
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

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

def detect_cat(img):
    cat_cascade = cv2.CascadeClassifier('initial_exploration/haarcascade_frontalcatface.xml')
    cat_ext_cascade = cv2.CascadeClassifier('initial_exploration/haarcascade_frontalcatface_extended.xml')
    eye_cascade = cv2.CascadeClassifier('initial_exploration/haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # This function returns tuple rectangle starting coordinates x,y, width, height
    SF = 1.05
    N = 6
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
    cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors = N)
    return (img, cats, cats_ext, eyes)

def create_data_for_model():
    """
    Creates the data points to put into the logistic regression model.
    """
    y_values = []
    files_seconds = []
    with open('labeled_results_test.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            #process all frames that have cats in them
            if(row[3] != 0):
                y_values.append(int(row[3]))
                files_seconds.append((row[1], float(row[2])))
    x_values = []
    for (filename,t) in files_seconds:
        cat=CatVideo("./cat_videos_1/"+filename)
        frame=cat.get_frame_time(t)
        #haarcascade = img, cats, cats_ext, eyes = detect_cat(frame)
        YOLO_frames = detect_raw_image(frame)
        eyes = []
        cats_ext = []
        for YOLO_frame in YOLO_frames:
            x1, y1, x2, y2, c, conf = YOLO_frame
            if (c==3):
                cats_ext.append(YOLO_frame)
            if (c==0):
                eyes.append(YOLO_frame)
        head_rat = 0
        eye_rat = 0
        confidence_head = 0
        sharpness = 0
        if len(cats_ext) > 0:
            print(t)
            print("Cat detected ^^")
            box_cleaned = [(cats_ext[0][2]-cats_ext[0][0], cats_ext[0][3]-cats_ext[0][1]), (frame.shape[0], frame.shape[1])]
            box = box_cleaned[0]
            frame_size = box_cleaned[1]
            box_area = box[0]*box[1]
            frame_area = frame_size[0]*frame_size[1]
            head_rat= box_area/frame_area
            confidence_head = cats_ext[0][5]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            sharpness = sharpness_score(gray_frame[cats_ext[0][1]:cats_ext[0][3],
                                                    cats_ext[0][0]:cats_ext[0][2]])


        if len(eyes) > 1:
            eye_dims = np.array([(eye[2]-eye[0])*(eye[3]-eye[1]) for eye in eyes])
            top_two_eyes = eye_dims[np.argsort(eye_dims)[-2:]]
            eye_rat = top_two_eyes[1]/top_two_eyes[0]

        x_values.append([eye_rat, head_rat, confidence_head, sharpness])
    return (x_values, y_values)

def return_weights(x, y):
    model = LogisticRegression(random_state=0).fit(x, y)
    # print(model.predict_proba(x))
    return(model.coef_)

def sharpness_cpbd_scores():
    """
    Creates the data points to put into the logistic regression model.
    """
    y_values = []
    files_seconds = []
    with open('labeled_sharpness.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            #process all frames that have cats in them
            if(row[3] != 0):
                y_values.append(int(row[3]))
                files_seconds.append((row[1], float(row[2])))
    sum_difference = 0
    counter = 0
    for i in range(len(files_seconds)):
        (filename,t) = files_seconds[i]
        sharp = y_values[i]
        cat=CatVideo("./cat_videos_1/"+filename)
        frame=cat.get_frame_time(t)
        YOLO_frames = detect_raw_image(frame)
        sharpness = 0
        cats_ext = []
        for YOLO_frame in YOLO_frames:
            x1, y1, x2, y2, c, conf = YOLO_frame
            if (c==3):
                cats_ext.append(YOLO_frame)
        if len(cats_ext) > 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            sharpness = cpbd.compute(gray_frame[cats_ext[0][1]:cats_ext[0][3],
                                                    cats_ext[0][0]:cats_ext[0][2]])
        if sharpness > 0:
            counter = counter + 1
            diff = np.abs(sharp - min(int(sharpness*5) + 1, 5))
            print(diff)
            sum_difference = sum_difference + diff
    return sum_difference/counter

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
    # x, y = create_data_for_model()
    # print(x)
    # print(y)
    # print(return_weights(x, y))
    # x = [[1.218487394957983, 0.0279296875, 0.7037841, 54.68026792890703], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.5206611570247934, 0, 0, 0], [1.0769230769230769, 0.04722029320987654, 0.9620437, 526.5716019847828], [0, 0, 0, 0], [2.0036363636363634, 0, 0, 0], [0, 0.0874291087962963, 0.8625808, 75.76298862573218], [0, 0, 0, 0], [0, 0.09327256944444444, 0.89233714, 74.4817941851501], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.321875, 0.04404272762345679, 0.57999426, 81.1821331524273], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2.2745098039215685, 0.034444444444444444, 0.90505916, 60.30154019563794], [1.4470443349753694, 0.02404128086419753, 0.5984081, 58.63572682079499], [0, 0.2752278645833333, 0.99680924, 93.21140057545018], [0, 0.026278935185185186, 0.3037922, 66.69662711699752], [0, 0.1733203125, 0.859682, 308.9636346178411], [1.5604422604422605, 0.148681640625, 0.98241955, 318.64434003633863], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2.9072079536039768, 0.053451967592592596, 0.3917927, 93.88059464812916], [1.5292353823088456, 0.07132908950617284, 0.44517136, 308.5772593182904], [0, 0, 0, 0], [1.0014814814814814, 0.034268904320987656, 0.73422855, 279.5494592947343], [0, 0, 0, 0], [1.5200666666666667, 0.3891373697916667, 0.9936548, 109.45998087311389], [0, 0, 0, 0], [1.148711169861203, 0.06390769675925925, 0.56578326, 166.67310621922078], [2.0616246498599438, 0.05586902006172839, 0.52870953, 45.698118124783335], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0.04068769290123457, 0.267209, 1081.9683617327057], [1.1893280632411067, 0.031080729166666668, 0.9459241, 67.5613177724024], [1.6732991014120668, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.1615094339622642, 0.038711419753086417, 0.5865378, 66.00797455831754], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.2249226690234203, 0.11200810185185185, 0.97693473, 132.29637081477108], [1.4325468844525107, 0.025862268518518517, 0.9246586, 78.11366757152967], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0.030017361111111113, 0.32484338, 46.48259199815469], [1.9118065433854907, 0.033449074074074076, 0.94556034, 1362.4285795513504], [1.0580793559516963, 0.03979166666666667, 0.7517149, 248.8354568434739], [0, 0, 0, 0], [0, 0, 0, 0], [1.3866666666666667, 0.030980902777777777, 0.9734631, 1008.2420843035129], [0, 0.03875, 0.30865678, 672.3903313448502], [0, 0.009887152777777778, 0.73675436, 3490.0165455685833], [0, 0, 0, 0], [0, 0, 0, 0], [1.1705955334987592, 0.014553433641975309, 0.8248493, 863.4024210191203], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.1929824561403508, 0.036631944444444446, 0.87088794, 151.97168841670225], [0, 0.031225887345679014, 0.93759686, 983.5388246200264], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.3148148148148149, 0.03673562885802469, 0.9501621, 90.72377947545125], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.080368906455863, 0.02529658564814815, 0.5183618, 427.7014197467314], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0.0428587962962963, 0.32413313, 823.8564165923634], [0, 0, 0, 0], [0, 0, 0, 0], [1.191311612364244, 0.11202739197530864, 0.976273, 213.3062745628741], [0, 0, 0, 0]]
    # y = [4, 3, 2, 2, 2, 5, 0, 4, 3, 3, 3, 4, 4, 2, 3, 3, 4, 3, 1, 4, 2, 3, 2, 4, 4, 2, 0, 2, 3, 4, 2, 4, 1, 4, 1, 2, 2, 2, 0, 1, 2, 1, 2, 2, 3, 3, 1, 2, 0, 2, 3, 3, 2, 1, 2, 3, 2, 3, 1, 1, 0, 3, 4, 4, 2, 2, 3, 1, 2, 0, 2, 4, 0, 1, 0, 3, 3, 1, 2, 0, 2, 1, 2, 2, 3, 0, 2, 2, 2, 2, 2, 0, 0, 2, 1, 3, 1, 3, 3, 2]
    # head_rats = [cats[1] if cats[1] < 0.15 else 0.15 for cats in x]
    # eye_rats = [cats[0] for cats in x]
    # confidence = [cats[2] for cats in x]
    # sharpness = [cats[3] for cats in x]
    # print(return_weights(x, y))
    print(sharpness_laplacian_scores())
    # plt.figure()
    # plt.title("Head Ratio vs. Rated Class")
    # plt.xlabel("Class")
    # plt.ylabel("Head Ratio")
    # plt.scatter(y, head_rats)
    # plt.show()
    # plt.figure()
    # plt.title("Ratio Between Eyes vs. Rated Class")
    # plt.xlabel("Class")
    # plt.ylabel("Ratio Between Eyes")
    # plt.scatter(y, eye_rats)
    # plt.show()
    # plt.figure()
    # plt.title("YOLO Confidence vs. Rated Class")
    # plt.xlabel("Class")
    # plt.ylabel("Confidence from YOLO Model")
    # plt.scatter(y, confidence)
    # plt.show()
    # plt.figure()
    # plt.title("Sharpness vs. Rated Class")
    # plt.xlabel("Class")
    # plt.ylabel("Sharpness (from Laplacian filter)")
    # plt.scatter(y, sharpness)
    # plt.show()
