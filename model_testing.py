"""
This script generates and populates a CSV file containing the outputs
of our baseline model and our developed model, which will be displayed in the CLI
so that we can compare the results.
"""

import os
import pandas as pd
from process_video import score_video, score_video_baseline, get_features, get_frame
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

directory = "./videos"
results_file = "./model_results.csv"

"""
Example of what CSV looks like:

video_name, video_number, frame_developed_model, frame_baseline_model
"cat1.mp4", 1, 34, 12
"""
rows_dict = pd.read_csv(results_file).to_dict(orient="list")


def ab_test(userid, model_csv, output_filename):
    test_results = list()
    model_df = pd.read_csv(model_csv)
    directory = "videos"
    def clicked_image(event, video_name, chosen_image):
        test_results.append("{},{},{}".format(userid, video_name, chosen_image))
        plt.close()

    for index, video in model_df.iterrows():
        if video["frame_developed_model"] == video["frame_baseline_model"]:
            continue
        # 0 corresponds to baseline shown on left; 1 corresponds to baseline shown on right
        random_side = np.random.randint(2)
        if random_side == 0:
            frame_left = video["frame_baseline_model"]
            frame_right = video["frame_developed_model"]
        else:
            frame_left = video["frame_developed_model"]
            frame_right = video["frame_baseline_model"]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
        plt.subplots_adjust(bottom=0.25)
        ax[0].imshow(get_frame(f"{directory}/{video['video_name']}", frame_left))
        ax[0].set_title("Image 1")
        ax[0].set_axis_off()

        button_1 = Button(plt.axes([0.20, 0.1, 0.15, 0.1]), "Image 1 is Better")
        button_1.on_clicked(
            lambda event: clicked_image(event, video["video_name"],
                                        "baseline_model" if random_side == 0 else "developed_model")
        )
        ax[1].imshow(get_frame(f"{directory}/{video['video_name']}", frame_right))
        ax[1].set_title("Image 2")
        ax[1].set_axis_off()

        button_2 = Button(plt.axes([0.65, 0.1, 0.15, 0.1]), "Image 2 is Better")
        button_2.on_clicked(
            lambda event: clicked_image(event, video["video_name"],
                                        "baseline_model" if random_side == 1 else "developed_model")
        )

        button_no_cat = Button(plt.axes([0.45, 0, 0.15, 0.1]), "No Cat in Either Image")
        button_no_cat.on_clicked(
            lambda event: clicked_image(event, video["video_name"], "no_cat")
        )
        fig.suptitle("Which Image is Better?")
        plt.show()

    with open(output_filename, "a") as output_file:
        output_file.write("\n".join(test_results) + "\n")


if __name__ == "__main__":
    print("Input user:")
    user = input()
    ab_test(user, "model_results.csv", "ab_test_results.txt")
