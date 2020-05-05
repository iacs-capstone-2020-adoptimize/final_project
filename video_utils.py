import cv2
import ffmpeg
import numpy as np
import re
from PIL import Image
import csv
import os

def get_frame(file, time, height=None, width=None):
    out, err = ffmpeg.input(file, ss=str(time), t=0).output(
        "pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1
    ).run(capture_stdout=True, capture_stderr=(height is None or width is None))
    if height is None or width is None:
        width, height = (int(val) for val in re.search(r"Video:.*, (\d*)x(\d*),", err.decode("utf-8")).groups())
    return np.frombuffer(out, np.uint8).reshape((height, width, 3))


def get_frame_opencv(filename, frame_number):
    capture = cv2.VideoCapture(filename)
    # I'm not sure why this doesn't work (or at least doesn't always work)
    # capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # status, frame = capture.read()
    for i in range(frame_number):
        capture.grab()
    return capture.retrieve()[1]


def get_all_frames_opencv(file):
    output = list()
    capture = cv2.VideoCapture(file)
    while True:
        playing, frame = capture.read()
        if not playing:
            break
        output.append(frame)
    return output


def get_all_frames_ffmpeg(file, vsync=0):
    probe = ffmpeg.probe(file)
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    out, err = ffmpeg.input(file, vsync=vsync, hide_banner=None).output(
        "pipe:", format="rawvideo", pix_fmt="rgb24", vf="showinfo"
    ).run(capture_stdout=True, capture_stderr=True)
    err = err.decode("utf-8")
    present_time = [float(entry.group()[len("pts_time:"):]) for entry in re.finditer(r"pts_time:\d*.\d*", err)]
    return present_time, np.frombuffer(out, np.uint8).reshape((-1, height, width, 3))


def get_pts_list(file, vsync=0):
    _, err = ffmpeg.input(file, vsync=vsync, hide_banner=None).output(
        "null", format="null", vf="showinfo"
    ).run(capture_stderr=True)
    err = err.decode("utf-8")
    pts_list = [float(entry.group()[len("pts_time:"):]) for entry in re.finditer(r"pts_time:\d*.\d*", err)]
    return pts_list


def iter_all_frames_ffmpeg(file, vsync=0):
    probe = ffmpeg.probe(file)
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    process = ffmpeg.input(file, vsync=vsync, hide_banner=None).output(
        "pipe:", format="rawvideo", pix_fmt="rgb24"
    ).run_async(pipe_stdout=True)
    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        else:
            yield np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))

def extract_frames_from_label_data(label_data_file_path):
    file = open(label_data_file_path)
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        video_name = row[1].split(".mp4")[0]
        frame_time = row[2]
        video = row[1]
        frame_time_norm = frame_time.replace(".", "_")
        frame_filepath = f"data/frames{video_name}_{frame_time_norm}.png"
        if os.path.exists(frame_file_path):
            continue
        frame = CatVideo(f"data/videos/{video}").get_frame_time(frame_time)
        frame_image = Image.fromarray(frame).save(frame_filepath)

class CatVideo:
    def __init__(self, file, vsync=0, loglevel="error"):
        self.file = file
        probe = ffmpeg.probe(file)
        video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        if video_stream is None:
            raise ValueError("No video stream found")
        self.width = int(video_stream["width"])
        self.height = int(video_stream["height"])
        if "tags" in video_stream and "rotate" in video_stream["tags"]:
            if video_stream["tags"]["rotate"] == "90" or video_stream["tags"]["rotate"] == "270":
                self.width, self.height = self.height, self.width
        self.duration = float(video_stream["duration"])
        self.vsync = vsync
        self.loglevel = loglevel
        self.pts_list = None

    def set_pts_list(self):
        _, err = ffmpeg.input(self.file, vsync=self.vsync, hide_banner=None).output(
            "null", format="null", vf="showinfo"
        ).run(capture_stderr=True)
        self.pts_list = [float(entry.group(1)) for entry in re.finditer(br"pts_time:([\d.]+) ", err)]

    def iter_all_frames(self):
        """
        Generator that yields each frame of the video.
        Please close as soon as you are done. Otherwise, there will be useless processes running until the garbage
        collector gets around to it.
        """
        process = ffmpeg.input(self.file, vsync=self.vsync, hide_banner=None, loglevel=self.loglevel).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24"
        ).run_async(pipe_stdout=True, pipe_stderr=(self.pts_list is None))
        try:
            while True:
                in_bytes = process.stdout.read(self.width * self.height * 3)
                if not in_bytes:
                    break
                else:
                    yield np.frombuffer(in_bytes, np.uint8).reshape((self.height, self.width, 3))
            if self.pts_list is None:
                _, err = process.communicate()
                if err is not None:
                    pts_found = list(re.finditer(br"pts_time:([\d.]+) ", err))
                    if len(pts_found) > 0:
                        pts_found = [float(match.group(1)) for match in match_list]
                        self.pts_list = pts_found
        except GeneratorExit as e:
            raise e  # I'm not familiar enough with generators to know whether this line is necessary
        finally:
            process.kill()

    def get_frame_num(self, frame):
        if self.pts_list is None:
            self.set_pts_list()
        frame_time = self.pts_list[frame]
        return self.get_frame_time(frame_time)

    def get_frame_time(self, frame_time):
        out, _ = ffmpeg.input(self.file, ss=str(frame_time), t=str(0.1), loglevel=self.loglevel).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1
        ).run(capture_stdout=True)
        raw_image = np.frombuffer(out, np.uint8)
        if raw_image.size == 0:
            raise ValueError("No frames found (video must have at least one frame with"
                             " presentation time stamp >= frame_time)")
        else:
            return raw_image.reshape((self.height, self.width, 3))

    def get_random_frame(self, seed=None, num_tries=10):
        if seed is not None:
            np.random.seed(seed)
        tries = 0
        while num_tries is None or tries < num_tries:
            tries += 1
            frame_time = np.random.random() * self.duration
            try:
                frame = self.get_frame_time(frame_time)
            except ValueError:
                pass
            else:
                return frame_time, frame
        return None


"""
Run this to grab all frames that we've scored.
"""
if __name__ == "__main__":
    extract_frames_from_label_data("data/regression_training/score_v2.csv")
