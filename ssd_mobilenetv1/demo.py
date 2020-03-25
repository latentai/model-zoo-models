import sys
import argparse
import numpy as np 
from models.ssd_mobilenet import ssd_300
import cv2
import numpy as np
from keras.optimizers import Adam
from misc.keras_ssd_loss import SSDLoss
from keras import backend as K
import os
import h5py
import keras
import time
from keras.preprocessing import image
from misc.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2


img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
swap_channels = True  # The color channel order in the original SSD is BGR
n_classes = 20  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
              1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
               1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_voc

aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [16, 32, 64, 100, 150, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

# 1: Build the Keras model

K.clear_session()  # Clear previous models from memory.

model = ssd_300(mode = 'training',
              image_size=(img_height, img_width, img_channels),
              n_classes=n_classes,
              l2_regularization=0.0005,
              scales=scales,
              aspect_ratios_per_layer=aspect_ratios,
              two_boxes_for_ar1=two_boxes_for_ar1,
              steps=steps,
              offsets=offsets,
              limit_boxes=limit_boxes,
              variances=variances,
              coords=coords,
              normalize_coords=normalize_coords,
              subtract_mean=subtract_mean,
              divide_by_stddev=None,
              swap_channels=swap_channels)


CLASSES = ['background',
             'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog',
             'horse', 'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor']


def predict(filename, weights):
    model.load_weights(weights)

    img = cv2.imread(filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.
    orig_images.append(img)

    ima = img
    image1 = cv2.resize(img,(300,300))
    image1 = np.array(image1,dtype=np.float32)

    image1 = image1[np.newaxis,:,:,:]
    input_images = np.array(image1)

    start_time = time.time()

    y_pred = model.predict(input_images)

    print("time taken by ssd", time.time() - start_time)

    y_pred_decoded = decode_y(y_pred,
                              confidence_thresh=0.25,
                              iou_threshold=0.45,
                              top_k=100,
                              input_coords='centroids',
                              normalize_coords=True,
                              img_height=img_height,
                              img_width=img_width)

    for box in y_pred_decoded[0]:
        print('Box:', box)

        xmin = int(box[-4] * orig_images[0].shape[1] / img_width)
        ymin = int(box[-3] * orig_images[0].shape[0] / img_height)
        xmax = int(box[-2] * orig_images[0].shape[1] / img_width)
        ymax = int(box[-1] * orig_images[0].shape[0] / img_height)

        print(xmin,xmax,ymin,ymax)
        cv2.rectangle(orig_images[0],(xmin, ymin), (xmax, ymax),(0,255,0),2)
        label = CLASSES[int(box[0])]
        cv2.putText(orig_images[0], label, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, .5, (200, 0, 0), 1)



    img = cv2.cvtColor(orig_images[0], cv2.COLOR_BGR2RGB)
    cv2.imwrite("predict.png", img)

    print('Prediction complete. See predict.png file.')

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--filename', type=str,
                        help='Path to file with image to make prediction on.')
    parser.add_argument('--weight_file',type=str,
                        help='weight file path')

    args = parser.parse_args()
    predict(args.filename, args.weight_file)
