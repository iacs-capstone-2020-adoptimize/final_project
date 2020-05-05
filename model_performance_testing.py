"""
This script generates and populates a CSV file containing the outputs
of our baseline model and our developed model, which will be displayed in the CLI
so that we can compare the results.
"""

import os
import pandas as pd
from video_utils import CatVideo
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np


"""
Example of what CSV looks like:

video_name, video_number, frame_developed_model, frame_baseline_model
"cat1.mp4", 1, 34, 12
"""


def ab_test(userid, baseline_model_csv, developed_model_csv, output_filename,
            directory="videos"):
    test_results = list()
    baseline_model = pd.read_csv(baseline_model_csv)
    developed_model = pd.read_csv(developed_model_csv)
    model_df = pd.merge(left=baseline_model, right=developed_model,
                        on="video_name", suffixes=("_baseline", "_developed"))
    def clicked_image(event, video_name, chosen_image):
        test_results.append("{},{},{}".format(userid, video_name, chosen_image))
        plt.close()

    for index, video in model_df.iterrows():
        if video["frame_developed"] == video["frame_baseline"]:
            continue
        # 0 corresponds to baseline shown on left; 1 corresponds to baseline shown on right
        random_side = np.random.randint(2)
        if random_side == 0:
            frame_left = video["frame_baseline"]
            frame_right = video["frame_developed"]
        else:
            frame_left = video["frame_developed"]
            frame_right = video["frame_baseline"]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
        plt.subplots_adjust(bottom=0.25)
        cat_video = CatVideo(f"{directory}/{video['video_name']}")
        ax[0].imshow(cat_video.get_frame_num(frame_left))
        ax[0].set_title("Image 1")
        ax[0].set_axis_off()

        button_1 = Button(plt.axes([0.20, 0.1, 0.15, 0.1]), "Image 1 is Better")
        button_1.on_clicked(
            lambda event: clicked_image(event, video["video_name"],
                                        "baseline" if random_side == 0 else "developed")
        )
        ax[1].imshow(cat_video.get_frame_num(frame_right))
        ax[1].set_title("Image 2")
        ax[1].set_axis_off()

        button_2 = Button(plt.axes([0.65, 0.1, 0.15, 0.1]), "Image 2 is Better")
        button_2.on_clicked(
            lambda event: clicked_image(event, video["video_name"],
                                        "baseline" if random_side == 1 else "developed")
        )

        button_no_cat = Button(plt.axes([0.45, 0, 0.15, 0.1]), "No Cat/Same Image")
        button_no_cat.on_clicked(
            lambda event: clicked_image(event, video["video_name"], "no_cat")
        )
        fig.suptitle("Which Image is Better?")
        plt.show()

    with open(output_filename, "a") as output_file:
        output_file.write("\n".join(test_results) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A/B Test Models")
    parser.add_argument("-user", default="")
    parser.add_argument("-baseline_model_results",
                        default="model_results/baseline_model.csv")
    parser.add_argument("-developed_model_results",
                        default="model_results/log_model.csv")
    parser.add_argument(
        "-video_dir", default="videos",
        help="Directory containing mp4 files to ask user to evaluate"
    )
    parser.add_argument("-output", default="ab_test_results.txt")
    args = parser.parse_args()
    ab_test(args.user, args.baseline_model_results,
            args.developed_model_results, args.output,
            directory=args.video_dir)
