"""
This script generates and populates a CSV file containing the outputs
of our baseline model and our developed model, which will be displayed in the CLI
so that we can compare the results.
"""

import os
import pandas as pd
from process_video import score_video, score_video_baseline, get_features
import re

directory = "./videos/30_videos"
results_file = "./model_results.csv"

"""
Example of what CSV looks like:

video_name, video_number, frame_developed_model, frame_baseline_model
"cat1.mp4", 1, 34, 12
"""
rows_df = pd.read_csv(results_file).to_dict()
print(rows_df)
import sys; sys.exit()


for filename in os.listdir(directory):
    print(rows_df)
    # print(filename)
    file_path = f"{directory}/{filename}"
    print(file_path)
    # processed_video = get_features(file_path)
    # developed_model_output = score_video(processed_video)
    # baseline_output = score_video_baseline(processed_video)

    developed_model_output = 1
    baseline_output = 0


    video_number = int(filename.split(".")[0][3:])

    #
    # new_row = {
    #     "video_name": filename,
    #     "video_number": video_number,
    #     "frame_developed_model": developed_model_output,
    #     "frame_baseline_model": baseline_output
    # }
    rows_df["video_name"].append(filename)
    rows_df["video_number"].append(video_number)
    rows_df["frame_developed_model"].append(frame_developed_model)
    rows_df["frame_baseline_model"].append(frame_baseline_model)
    print(rows_df)
    rows_df.to_csv(results_file)
    print("Saved to CSV", filename)
