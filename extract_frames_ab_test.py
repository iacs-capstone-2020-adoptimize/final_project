import os
import pandas as pd
from process_video import score_video, score_video_baseline, get_features, get_frame
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

directory = "data/videos"
results_file = "ab_testing/frames_to_test/model_results.csv"

"""
Example of what CSV looks like:

video_name, video_number, frame_developed_model, frame_baseline_model
"cat1.mp4", 1, 34, 12
"""

rows_df = pd.read_csv(results_file)

for filename in os.listdir(directory):
    if filename[-4:].lower() == ".mp4" and filename not in rows_df["video_name"].values:
        new_row = dict()
        # print(rows_dict)
        # print(filename)
        file_path = f"{directory}/{filename}"
        print(file_path)
        try:
            processed_video = get_features(file_path, sample_rate=1)
            developed_model_output = score_video(processed_video)
            baseline_output = score_video_baseline(processed_video)
        except ValueError:
            continue

        video_number = int(filename.split(".")[0][3:])

        rows_df = rows_df.append({
            "video_name": filename, "frame_baseline_model": baseline_output,
            "frame_developed_model": developed_model_output
        }, ignore_index=True)
        rows_df.to_csv(results_file, index=False)
        print("Saved to CSV", filename)
