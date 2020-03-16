# TrainYourOwnYOLO: Inference
In this step, we test our detector on cat and dog images and videos located in [`TrainYourOwnYOLO/Data/Source_Images/Test_Images`](/Data/Source_Images/Test_Images). If you like to test the detector on your own images or videos, place them in the [`Test_Images`](/Data/Source_Images/Test_Images) folder. 

## Testing Your Detector
To detect objects run the detector script from within the [`TrainYourOwnYOLO/3_Inference`](/3_Inference/) directory: 
```
python Detector.py
```
Note that by default the model is assumed to be in [Model_Weights/trained_weights_final.h5](/yolo_training/Data/Model_Weights/trained_weights_final.h5)
The outputs are saved to [`Source_Images/Test_Image_Detection_Results`](/yolo_training/Data/Source_Images/Test_Image_Detection_Results). The outputs include the original images with bounding boxes and confidence scores as well as a file called [`Detection_Results.csv`](/yolo_training/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv) containing the image file paths and the bounding box coordinates. For videos, the output files are videos with bounding boxes and confidence scores. To list available command line options run `python Detector.py -h`.

### That's all!
Congratulations on building your own custom YOLOv3 computer vision application.

I hope you enjoyed this tutorial and I hope it helped you get our own computer vision project off the ground:

- Please **star** ⭐ this repo to get notifications on future improvements and
- Please **fork** 🍴 this repo if you like to use it as part of your own project.
