#!/usr/bin/python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import yaml
import pickle
import os
import argparse
from ssd_layers import PriorBox
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']


def restore_tf_checkpoint(conf, sess):

    sess.run(tf.compat.v1.initialize_all_variables())
    model = tf.keras.models.load_model(conf['tf_model_path'] + '/trained.h5',
                               custom_objects={
                                   'PriorBox': PriorBox,
                                   'compute_loss': MultiboxLoss(21, neg_pos_ratio=2.0).compute_loss
                               })

    # print('tf version: {}'.format(tf.__version__))
    # # predictions_target
    # sess.run(tf.compat.v1.initialize_all_variables())
    # tf_meta_path = glob('{}/*.meta'.format(conf['tf_model_path']))[0]
    # saver = tf.compat.v1.train.import_meta_graph(tf_meta_path)
    # saver.restore(sess, tf.compat.v1.train.latest_checkpoint(conf['tf_model_path']))
    # graph = tf.compat.v1.get_default_graph()
    #
    # input_placeholder = graph.get_tensor_by_name(conf['input_node'])
    # output_placeholder = graph.get_tensor_by_name(conf['output_node'])
    #
    # return {
    #     'sess': sess,
    #     'in': input_placeholder,
    #     'out': output_placeholder
    # }
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_settings', help='Path to settings')
    parser.add_argument('--path_to_model', help='Path to model')
    parser.add_argument('--path_to_demo_img', help='Path to model', required=True)

    args = parser.parse_args()
    print(f'[II] sets file: {args.path_to_settings}')
    with open(args.path_to_settings, 'r') as fp:
        sets = yaml.safe_load(fp)

    np.set_printoptions(suppress=True)

    config = tf.compat.v1.ConfigProto()
    with tf.compat.v1.Session(config=config) as s:
        tf_inference = restore_tf_checkpoint(sets, s)
        inputs = []
        images = []
        img_path = args.path_to_demo_img

        img = image.load_img(img_path, target_size=(sets['img_height'], sets['img_width']))
        img = image.img_to_array(img)

        images.append(img)
        inputs.append(img.copy())

        inputs = preprocess_input(np.array(inputs))
        dir_path = os.path.dirname(os.path.realpath(__file__))


        priors = pickle.load(open(os.path.join(dir_path, 'priorFiles/prior_boxes_ssd300MobileNetV2_224_224.pkl'), 'rb'))

        bbox_util = BBoxUtility(21, priors)

        preds = tf_inference.predict(inputs)
        results = bbox_util.detection_out(preds)


        for i, img in enumerate(images):
            # Parse the outputs.
            det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]

            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

            plt.imshow(img / 255.)
            currentAxis = plt.gca()
            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * img.shape[1]))
                ymin = int(round(top_ymin[i] * img.shape[0]))
                xmax = int(round(top_xmax[i] * img.shape[1]))
                ymax = int(round(top_ymax[i] * img.shape[0]))
                score = top_conf[i]
                label = int(top_label_indices[i])
                label_name = voc_classes[label - 1]
                display_txt = '{:0.2f}, {}'.format(score, label_name)
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                color = colors[label]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

            result_file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/result.png')
            plt.savefig(result_file_name)
    print('\n\n\nResult file is stored to : {}\n\n'.format(result_file_name))