# TrainYourOwnYOLO: Training

## Training
We used the CAT dataset available here[https://www.kaggle.com/crawford/cat-dataset] or here[https://archive.org/details/CAT_DATASET]. (Careful that CAT_00/00000003_015.jpg and CAT_00/00000003_019.jpg.cat should be ignored as the first is missing the label and the second is missing the image.) Move the data files so that CAT_00 through CAST_06 are in the [Source_Images](/yolo_training/Data/Source_Images) directory. Run `extract_feature_box.py` to create the annotation file.

## Download and Convert Pre-Trained Weights
Before getting started download the pre-trained YOLOv3 weights and convert them to the keras format. To run both steps run the download and conversion script from within the [`TrainYourOwnYOLO/2_Training`](/2_Training/) directory:

```
python Download_and_Convert_YOLO_weights.py
```

The weights are pre-trained on the [ImageNet 1000 dataset](http://image-net.org/challenges/LSVRC/2015/index) and thus work well for object detection tasks that are very similar to the types of images and objects in the ImageNet 1000 dataset.

## Train YOLOv3 Detector
To start the training, run the training script from within the [`TrainYourOwnYOLO/2_Training`](/2_Training/) directory:
```
python Train_YOLO.py 
```
Depending on your set-up, this process can take a few minutes to a few hours. The final weights are saved in [`TrainYourOwnYOLO/Data/Model_weights`](/Data/Model_weights). To list available command line options run `python Train_YOLO.py -h`.

If training is too slow on your local machine, consider using cloud computing services such as AWS to speed things up. To learn more about training on AWS navigate to [`TrainYourOwnYOLO/2_Training/AWS`](/2_Training/AWS).

### That's all for training! 
Next, go to [`TrainYourOwnYOLO/3_Inference`](/3_Inference) to test your YOLO detector on new images!
