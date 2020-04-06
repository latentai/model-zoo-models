import os
import cv2
import sys
import time
import shutil
import argparse

import numpy as np

from glob import glob
from keras import backend as K
from misc.ssd_box_encode_decode_utils import decode_y
from models.ssd_mobilenet import ssd_300

import xml.etree.ElementTree as ET
from eval.BoundingBox import BoundingBox as BoundingBox
from eval.BoundingBoxes import BoundingBoxes as BoundingBoxes
from eval.Evaluator import *
from eval.utils import BBFormat as BBFormat
from eval.utils import CoordinatesType

VERSION = '0.1 (beta) modified'

currentPath = os.path.dirname(os.path.abspath(__file__))

# ------------- Prepare detections ----------------

def generate_detections_for_ssd(images_dir, weights):

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


    CLASSES = ['Background',
                 'Aeroplane', 'Bicycle', 'Bird', 'Boat',
                 'Bottle', 'Bus', 'Car', 'Cat',
                 'Chair', 'Cow', 'Diningtable', 'Dog',
                 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                 'Sheep', 'Sofa', 'Train', 'Tvmonitor']


    def predict(model, img_path):
        """
        Predict on single image for now.

        Returns list [<class_name> <left> <top> <right> <bottom>]
        """

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_images = []  # Store the images here.
        input_images = []  # Store resized versions of the images here.
        orig_images.append(img)

        image1 = cv2.resize(img, (300, 300))
        image1 = np.array(image1, dtype=np.float32)

        image1 = image1[np.newaxis, :, :, :]
        input_images = np.array(image1)

        start_time = time.time()

        y_pred = model.predict(input_images)

        print("time taken by ssd", time.time() - start_time)

        # get [class_id, confidence, xmin, ymin, xmax, ymax]
        y_pred_decoded = decode_y(y_pred,
                                  confidence_thresh=0.25,
                                  iou_threshold=0.45,
                                  top_k=100,
                                  input_coords='centroids',
                                  normalize_coords=True,
                                  img_height=img_height,
                                  img_width=img_width)

        y_pred_decoded = y_pred_decoded[0]

        return [
            [CLASSES[int(y_pred[0])]] + [x / 300 for x in y_pred[1:]]
            for y_pred
            in y_pred_decoded
        ]


    def generate_detections(images_dir, weights):
        model.load_weights(weights)

        if not os.path.exists('detections'):
            os.mkdir('detections')

        imgs_paths = glob(images_dir + '/*.jpg')

        for path in imgs_paths:
            y_pred = predict(model, path)
            print(len(y_pred), [y[0] for y in y_pred])

            file_name = os.path.split(path)[1].split('.')[0] + '.txt'
            file_path = os.path.join('detections', file_name)

            with open(file_path, 'w') as fp:
                for y in y_pred:
                    fp.write(' '.join([str(x) for x in y]) + '\n')


    generate_detections(images_dir, weights)

    print('Preparation complete.')

# ------------- Prepare detections end section ----------------

# -------------------- Perform evaluation ---------------------

'''
Object Detection Metrics - Pascal VOC

This project applies the most popular metrics used to evaluate object detection
algorithms. The current implemention runs the Pascal VOC metrics.

For further references, please check: https://github.com/rafaelpadilla/Object-Detection-Metrics

Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)
'''

# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg

def extract_annotation_from_xml(path):
    xml = ET.parse(path)
    xml_root = xml.getroot()
    tags = {'size': None, 'objects': list()}
    for child in xml_root:
        if child.tag == 'size':
            tags['size'] = child
        if child.tag == 'object':
            tags['objects'].append(child)

    size = {
        'width': float(tags['size'].find('width').text),
        'height': float(tags['size'].find('height').text)
    }

    tags['size'] = size

    for i in range(len(tags['objects'])):
        bbox = {
            'name': tags['objects'][i].find('name').text.capitalize(),
            'xmin': float(tags['objects'][i].find('bndbox').find('xmin').text) / tags['size']['width'],
            'ymin': float(tags['objects'][i].find('bndbox').find('ymin').text) / tags['size']['height'],
            'xmax': float(tags['objects'][i].find('bndbox').find('xmax').text) / tags['size']['width'],
            'ymax': float(tags['objects'][i].find('bndbox').find('ymax').text) / tags['size']['height'],
        }

        tags['objects'][i] = bbox

    return tags

def getBoundingBoxesXML(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """
    Read xml files containing bounding boxes (ground truth and detections) in VOC format.
    It only supports format: <class_name> <left> <top> <right> <bottom>
    """
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob("*.xml")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        ann = extract_annotation_from_xml(f)

        nameOfImage = f.replace(".xml", "")

        for obj in ann['objects']:
            idClass = obj['name']
            x = obj['xmin']
            y = obj['ymin']
            w = obj['xmax']
            h = obj['ymax']
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.GroundTruth,
                format = bbFormat)
        allBoundingBoxes.addBoundingBox(bb)
        if idClass not in allClasses:
            allClasses.append(idClass)

    return allBoundingBoxes, allClasses



def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


def evaluate(args):
    # Get current path to set default folders


    iouThreshold = args.iouThreshold

    # Arguments validation
    errors = []
    # Validate formats
    gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
    detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
    # Groundtruth folder
    if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
        gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
    else:
        # errors.pop()
        gtFolder = os.path.join(currentPath, 'groundtruths')
        if os.path.isdir(gtFolder) is False:
            errors.append('folder %s not found' % gtFolder)
    # Coordinates types
    gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
    detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
    imgSize = (0, 0)
    if gtCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
    if detCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
    # Detection folder
    if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
        detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
    else:
        # errors.pop()
        detFolder = os.path.join(currentPath, 'detections')
        if os.path.isdir(detFolder) is False:
            errors.append('folder %s not found' % detFolder)
    if args.savePath is not None:
        savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
    else:
        savePath = os.path.join(currentPath, 'results')
    # Validate savePath
    # If error, show error messages
    if len(errors) != 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()

    # Create directory to save results
    shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
    os.makedirs(savePath)
    # Show plot during execution
    showPlot = args.showPlot

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxesXML(
        gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=savePath,
        showGraphic=showPlot)

    f = open(os.path.join(savePath, 'results.txt'), 'w')
    f.write('Object Detection Metrics\n')
    f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % cl)
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)
# -------------- Perform evaluation end section ---------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--images_dir', type=str,
                        help='Path to the directory with images to make prediction on.')
    parser.add_argument('--weight_file', type=str,
                        help='weight file path')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    # Mandatory
    parser.add_argument(
        '-gt',
        '--gtfolder',
        dest='gtFolder',
        default=os.path.join(currentPath, 'groundtruths'),
        metavar='',
        help='folder containing your ground truth bounding boxes')
    parser.add_argument(
        '-det',
        '--detfolder',
        dest='detFolder',
        default=os.path.join(currentPath, 'detections'),
        metavar='',
        help='folder containing your detected bounding boxes')
    # Optional
    parser.add_argument(
        '-t',
        '--threshold',
        dest='iouThreshold',
        type=float,
        default=0.5,
        metavar='',
        help='IOU threshold. Default 0.5')
    parser.add_argument(
        '-gtformat',
        dest='gtFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the ground truth bounding boxes: '
        '(\'xywh\': <left> <top> <width> <height>)'
        ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-detformat',
        dest='detFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the detected bounding boxes '
        '(\'xywh\': <left> <top> <width> <height>) '
        'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-gtcoords',
        dest='gtCoordinates',
        default='abs',
        metavar='',
        help='reference of the ground truth bounding box coordinates: absolute '
        'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-detcoords',
        default='abs',
        dest='detCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: '
        'absolute values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-imgsize',
        dest='imgSize',
        metavar='',
        help='image size. Required if -gtcoords or -detcoords are \'rel\'')
    parser.add_argument(
        '-sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
    parser.add_argument(
        '-np',
        '--noplot',
        dest='showPlot',
        action='store_false',
        help='no plot is shown during execution')

    args = parser.parse_args()

    generate_detections_for_ssd(args.images_dir, args.weight_file)
    evaluate(args)
