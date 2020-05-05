import sys
sys.path.append('../')

from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
from video_utils import CatVideo
from yolo_training.Detector import detect_raw_image

# model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

# model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset
#
model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset


test_frame = CatVideo("../videos/cat2.mp4").get_frame_time(11)
# load any of the 3 pretrained models



out = model.predict_segmentation(
    inp=test_frame,
    out_fname="test_output/segmentation_test.png"
)

import pdb; pdb.set_trace()

print("Hello")
