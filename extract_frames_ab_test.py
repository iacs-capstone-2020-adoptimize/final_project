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

for filename in os.listdir(directory):
    if filename[-4:].lower() == ".mp4" and filename not in rows_dict["video_name"]:
        # print(rows_dict)
        # print(filename)
        file_path = f"{directory}/{filename}"
        print(file_path)
        try:
            processed_video = get_features(file_path)
            developed_model_output = score_video(processed_video)
            baseline_output = score_video_baseline(processed_video)
        except ValueError:
            continue

        video_number = int(filename.split(".")[0][3:])

        rows_dict["video_name"].append(filename)
        rows_dict["video_number"].append(video_number)
        rows_dict["frame_developed_model"].append(developed_model_output)
        rows_dict["frame_baseline_model"].append(baseline_output)
        rows_df = pd.DataFrame(rows_dict)
        rows_df.to_csv(results_file, index=False)
        print("Saved to CSV", filename)
