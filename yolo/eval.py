import json
import os

import cv2
from detections_utils.eval import evaluate
from detections_utils.eval import parser
from detections_utils.voc_subset_10_percent import voc_subset
from yolo3.utils import get_yolo_boxes
from yolo3.yolo_as_tf import load_model_tf

from demo import restore_keras_checkpoint


def prepare_detections(args):
    config_path  = args.conf
    input_path   = args.input
    det_folder = args.detFolder
    tf_checkpoint_dir = args.tf_checkpoint_dir
    binary_dir = args.binary_dir
    use_subset = args.use_subset
    is_dequantize = args.dequantize

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    if tf_checkpoint_dir is not None:
        infer_model = load_model_tf(tf_checkpoint_dir)
    elif binary_dir is not None:
        from yolo3.x86model import Model as X86Model
        infer_model = X86Model(dequantize=is_dequantize)
        infer_model.load(binary_dir)
    else:
        infer_model = restore_keras_checkpoint(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    ###############################
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    if use_subset:
        voc = [x.split('.')[0] for x in voc_subset]
        image_paths = [
            x
            for x
            in image_paths
            if os.path.split(x)[1].split('.')[0] in voc
        ]

    print(f'Number of images to do detections on: {len(image_paths)}')

    if not os.path.exists(det_folder):
        os.mkdir(det_folder)

    # the main loop
    for image_path in image_paths:
        image = cv2.imread(image_path)
        print(image.shape, image_path)

        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        filtered_boxes = list()
        for box in boxes:
            if box.get_score() > 0.0:
                filtered_boxes.append([config['model']['labels'][box.as_list()[0]].title()] + box.as_list()[1:])

        print(len(filtered_boxes), filtered_boxes)
        print('-' * 20)
        # <class_name> <left> <top> <right> <bottom>
        preds_file = os.path.split(image_path)[1].split('.')[0] + '.txt'
        h, w = image.shape[:2]
        with open(os.path.join(det_folder, preds_file), 'w') as fp:
            for box in filtered_boxes:
                class_name, left, top, right, bottom = box
                left, top, right, bottom = left / w, top / h, right / w, bottom / h

                fp.writelines(f'{class_name} {left} {top} {right} {bottom}\n')


if __name__ == '__main__':
    parser.add_argument('-c', '--conf', help='Path to configuration file.')
    parser.add_argument('-i', '--input', help='Path to a directory with images.')
    parser.add_argument('-u', '--use_cache', default=True, help='Whether to user previously generated detetions or not.')
    parser.add_argument('-tf', '--tf_checkpoint_dir', help='path to tensorflow checkpoint directory')
    parser.add_argument('-b', '--binary_dir', help='path to compiled checkpoint directory')
    parser.add_argument('-s', '--use_subset', default=False, help='True\False: whether to user small VOC subset (10%) or not.')
    parser.add_argument('-d', '--dequantize', default=False, help='True\False: whether to dequantize comiled model. Used for int8 models.')

    args = parser.parse_args()
    args.detFolder = os.path.abspath(args.detFolder)
    print('\n\n')
    print(args)
    print('\n\n')

    if not args.use_cache or not os.path.exists(args.detFolder):
        print('Preparing detections...')
        prepare_detections(args)

    print('Evaulating...')
    evaluate(args)
    print('Evaluation complete.')
