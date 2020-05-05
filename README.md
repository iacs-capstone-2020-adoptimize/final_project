# Overview
This repository contain the research and implementation for finding the "best" frame of a shelter cat in a recorded video. Finding the optimal image helps increase the cat's chances of getting adopted, thus reducing its likelihood of euthanasia.

This project is a partnership between Harvard IACS and Adoptimize + APA.

# Presentations
All presentations can be found under the "submissions" folder.

For the ignite talk, while the PDF file is uploaded to this repo, see [this link](https://docs.google.com/presentation/d/e/2PACX-1vRrVxcdSibALxJl2l-i1bayLf8kuZ0f5XR1zhvXkTJ_5pqAfzxpP9xV7iwFLLYelSUk5gnVV4fnSasJ/pub?start=false&loop=false&delayms=10000&fbclid=IwAR3-cCFefDrCtT9wZmlVh4v2k72EULOf3LCDs-pES9IL6NJqcT57k_g39Ro&slide=id.g7dfd9a6152_2_0) for the slides with built-in timer.

# General Setup
1. Clone this repository
2. Create a directory in the root called `videos`. This is git-ignored to prevent adding large files into our git history, which would otherwise bloat our repository. Add all cat videos into this folder. They should be named with the convention "cat{number}.mp4".
3. Run `pip install -r requirements.txt`. This should download all requirements necessary to run any script in this repository's subdirectories.

# Scripts

# Data
Outputs and data for any of our models or scripts can be found here. The `ab_testing` folder contains all of the data extracted from `extract_frames_ab_test.py`

# YOLO
We're currently looking into using the YOLO model for object detection, training it to detect a cat's eyes, ears, and mouth.

We are using Anton Muehlemann's wrapper around YOLO (with tools to make training as easy as possible), "TrainYourOwnYOLO". Note that we've copied over his code into this repo under the `yolo_training` folder, and adapted it to our own purposes. (Namely the default in Yrain_YOLO.py). For a direct link to Anton's original repository, check out this link: https://github.com/AntonMu/TrainYourOwnYOLO

### YOLO setup
To run our YOLO model, install the model weights located in our Google Drive, and save them to the following path: yolo_training/Data/Model_Weights/trained_weights_final.h5
