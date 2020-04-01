import os
import numpy as np
from matplotlib.widgets import Button
from utils import CatVideo
from matplotlib import pyplot as plt



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Label some photos")
    parser.add_argument("-user", default="")
    parser.add_argument("-seed", type=int)
    parser.add_argument("-input", default="videos", help="Directory containing mp4 files to ask user to evaluate")
    parser.add_argument("-output", default="labeled_results.csv")
    args = parser.parse_args()

    np.random.seed(args.seed)
    video_list = [file for file in os.listdir(args.input) if file[-4:].lower() == ".mp4"]
    if len(video_list) > 0:
        with open(args.output, "a") as output_file:
            testing = True
            def clicked_image(event, file, frame_time, response):
                output_file.write(",".join((args.user, file, str(frame_time), str(response))) + "\n")
                plt.close()

            def quit_testing(event):
                global testing
                testing = False
                plt.close()

            while testing:
                test_video_file = video_list[np.random.randint(len(video_list))]
                test_video = CatVideo(os.path.join(args.input, test_video_file))
                time, frame = test_video.get_random_frame()
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.subplots_adjust(bottom=0.2)
                ax.imshow(frame)
                ax.set_title("Image")
                ax.set_axis_off()
                fig.suptitle("How strongly do you agree with the following statement?\nI am more likely to adopt"
                             + " this cat after seeing this photo.")
                button_arr = list()
                for i, text in enumerate(["Strongly Disagree", "Disagree", "Neither Agree\nnor Disagree", "Agree",
                                          "Strongly Agree"]):
                    button_arr.append(Button(plt.axes([0.05 + 0.19 * i, 0.1, 0.14, 0.1]), text))
                    button_arr[-1].on_clicked(lambda event: clicked_image(event, test_video_file, time, i + 1))
                quit_button = Button(plt.axes([0.02, 0.02, 0.1, 0.05]), "Quit")
                quit_button.on_clicked(quit_testing)
                plt.show()
    else:
        raise ValueError("No mp4 files found")