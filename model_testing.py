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




# rows_dict = {"video_name": [], "video_number": [], "frame_developed_model": [], "frame_baseline_model": []}
rows_dict = pd.read_csv(results_file).to_dict(orient="list")


# for filename in os.listdir(directory):
#     if filename[-4:].lower() == ".mp4" and filename not in rows_dict["video_name"]:
#         # print(rows_dict)
#         # print(filename)
#         file_path = f"{directory}/{filename}"
#         print(file_path)
#         try:
#             processed_video = get_features(file_path)
#             developed_model_output = score_video(processed_video)
#             baseline_output = score_video_baseline(processed_video)
#         except ValueError:
#             continue
#
#         video_number = int(filename.split(".")[0][3:])
#
#         rows_dict["video_name"].append(filename)
#         rows_dict["video_number"].append(video_number)
#         rows_dict["frame_developed_model"].append(developed_model_output)
#         rows_dict["frame_baseline_model"].append(baseline_output)
#         rows_df = pd.DataFrame(rows_dict)
#         rows_df.to_csv(results_file, index=False)
#         print("Saved to CSV", filename)


def ab_test(userid, model_csv, output_filename):
    test_results = list()
    model_df = pd.read_csv(model_csv)
    directory = "videos"
    def clicked_image(event, chosen_image):
        test_results.append("{},{}".format(userid, chosen_image))
        plt.close()
    for index, video in model_df.iterrows():
        if video["frame_developed_model"] == video["frame_baseline_model"]:
            continue
        # 0 corresponds to developed shown on left; 1 corresponds to developed shown on right
        random_side = np.random.randint(2)
        if random_side == 0:
            frame_left = video["frame_developed_model"]
            frame_right = video["frame_baseline_model"]
        else:
            frame_left = video["frame_baseline_model"]
            frame_right = video["frame_developed_model"]
        fig, ax = plt.subplots(3, 2, figsize=(12.8, 4.8))
        ax[0, 0].imshow(get_frame(f"{directory}/{video['video_name']}", frame_left))
        ax[0, 0].set_title("Image 1")
        ax[0, 0].set_axis_off()
        button_1 = Button(ax[1, 0], "Image 1 is Better")
        button_1.on_clicked(
            lambda event: clicked_image(event, "developed_model" if random_side == 0 else "baseline_model")
        )
        ax[0, 1].imshow(get_frame(f"{directory}/{video['video_name']}", frame_right))
        ax[0, 1].set_title("Image 2")
        ax[0, 1].set_axis_off()
        button_2 = Button(ax[1, 1], "Image 2 is Better")
        button_2.on_clicked(
            lambda event: clicked_image(event, "developed_model" if random_side == 1 else "baseline_model")
        )
        button_no_cat = Button(ax[2, 0], "No Cat in Either Image")
        button_no_cat.on_clicked(
            lambda event: clicked_image(event, "no_cat")
        )
        fig.suptitle("Which Image is Better?")
        plt.show()
    with open(output_filename, "a") as output_file:
        output_file.write("\n".join(test_results))

if __name__ == "__main__":
    ab_test("", "model_results.csv", "ab_test_results.txt")