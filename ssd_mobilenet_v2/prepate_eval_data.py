import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import yaml
import pickle
import numpy as np
import os
from glob import glob
import shutil
from xml.etree import ElementTree
from tqdm import tqdm
from model.ssd300MobileNetV2Lite import SSD
from keras.preprocessing import image
from scipy.misc import imread
from ssd_utils import BBoxUtility
from keras.applications.imagenet_utils import preprocess_input


def restore_tf_checkpoint(conf, sess):
    print('tf version: {}'.format(tf.__version__))
    # predictions_target
    sess.run(tf.compat.v1.initialize_all_variables())
    tf_meta_path = glob('{}/*.meta'.format(conf['tf_model_path']))[0]
    saver = tf.compat.v1.train.import_meta_graph(tf_meta_path)
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(conf['tf_model_path']))
    graph = tf.compat.v1.get_default_graph()

    input_placeholder = graph.get_tensor_by_name(conf['input_node'])
    output_placeholder = graph.get_tensor_by_name(conf['output_node'])

    return {
        'sess': sess,
        'in': input_placeholder,
        'out': output_placeholder
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_settings', help='Path to dataset', required=True)
    parser.add_argument('--model_checkpoints', help='Model checkpoints', required=True)
    parser.add_argument('--n_images', help='Model checkpoints', default=4000)

    NUM_CLASSES = 21
    THRESHOLD = 0.6
    CLASSES = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']

    args = parser.parse_args()

    with open(args.path_to_settings, 'r') as fp:
        sets = yaml.safe_load(fp)

    input_shape = (sets['img_height'], sets['img_width'], 3)

    priors = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           'priorFiles/prior_boxes_ssd300MobileNetV2.pkl'), 'rb'))

    np.set_printoptions(suppress=True)

    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    config = tf.compat.v1.ConfigProto()

    inputs = []
    images = []
    result_detections = []
    result_images = []
    annotation_files = []

    print('Prepare : {} files for evaluation. '.format(args.n_images))

    with open(os.path.join(sets['dataset_dir'], 'VOC2007/ImageSets/Main/test.txt'), 'r') as annot_f:
        for annotation in tqdm(list(annot_f)[:args.n_images]):
            try:
                img_path = os.path.join(sets['dataset_dir'], 'VOC2007/JPEGImages/') + annotation.split(' ')[0].strip() + '.jpg'
                img = image.load_img(img_path, target_size=(300, 300))
                img = image.img_to_array(img)
                result_images.append(img_path)
                images.append(imread(img_path))
                inputs.append(img.copy())
                annotation_files.append(annotation)
            except Exception as e:
                print('Error while opening file.', e)

    with tf.compat.v1.Session(config=config) as s:
        tf_inference = restore_tf_checkpoint(sets, s)
        inputs = preprocess_input(np.array(inputs))
        img_per_batch = 5
        results = []
        start_index = 0

        for end_index in tqdm(range(img_per_batch, inputs.shape[0] + 1, img_per_batch)):
            preds = tf_inference['sess'].run(fetches=tf_inference['out'], feed_dict={
                tf_inference['in']: inputs[start_index:end_index, :]
            })
            results.extend(bbox_util.detection_out(preds))
            start_index = end_index

        for i, img in tqdm(enumerate(images)):
            # Parse the outputs.
            det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]

            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= THRESHOLD]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            detections = []
            for i in range(top_conf.shape[0]):
                '''
                xmin = int(round(top_xmin[i] * img.shape[1]))
                ymin = int(round(top_ymin[i] * img.shape[0]))
                xmax = int(round(top_xmax[i] * img.shape[1]))
                ymax = int(round(top_ymax[i] * img.shape[0]))
                '''
                xmin = top_xmin[i]
                ymin = top_ymin[i]
                xmax = top_xmax[i]
                ymax = top_ymax[i]

                score = top_conf[i]
                label = int(top_label_indices[i])
                label_name = CLASSES[label - 1]
                detections.append(
                    ['{:.2f}'.format(xmin), '{:.2f}'.format(ymin), '{:.2f}'.format(xmax), '{:.2f}'.format(ymax), label_name,
                     '{:.2f}'.format(score)])
            result_detections.append(detections)

        print('Test images: {}'.format(len(result_images)))

        model_predictions = []
        MODEL_PREDICTION_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_evaluation/model_prediction/')
        predicted_images = []

        for index, image_filename in tqdm(enumerate(result_images)):
            image_name = os.path.basename(image_filename)
            path_elements = image_name[:-4]
            predicted_images.append(image_name[:-4])
            annot_dir = os.path.join(MODEL_PREDICTION_PATH)
            os.makedirs(annot_dir, exist_ok=True)
            annot_name = '{}.txt'.format(path_elements)
            annot_filename = os.path.join(annot_dir, annot_name)
            with open(annot_filename, 'w') as output_f:
                for d in result_detections[index]:
                    left, top, right, botton, classe, score = d[0], d[1], d[2], d[3], d[4], d[5]
                    model_predictions.append((classe, score, left, top, right, botton))
                    output_f.write('{} {} {} {} {} {}\n'.format(classe, score, left, top, right, botton))

        GROUND_TRUTH_LABELS = os.path.join(sets['dataset_dir'], 'VOC2007/Annotations')
        GROUND_TRUTH_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_evaluation/ground_truth/')

        for f in glob(GROUND_TRUTH_PATH + '*'):
            os.remove(f)
        filenames = os.listdir(GROUND_TRUTH_LABELS)
        ground_images = []

        for filename in tqdm(filenames):
            if filename[:-4] not in predicted_images:
                continue
            ground_images.append(filename[:-4])
            tree = ElementTree.parse(os.path.join(GROUND_TRUTH_LABELS + '/{}'.format(filename)))
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text) / width
                    ymin = float(bounding_box.find('ymin').text) / height
                    xmax = float(bounding_box.find('xmax').text) / width
                    ymax = float(bounding_box.find('ymax').text) / height
                    class_name = object_tree.find('name').text.title()
                bounding_box = [class_name, xmin, ymin, xmax, ymax]
                bounding_boxes.append(bounding_box)

            with open(os.path.join(GROUND_TRUTH_PATH, filename.replace('xml', 'txt')), 'w+') as f:
                for p in bounding_boxes:
                    f.write(' '.join([str(s) for s in p]) + "\n")

        print('Completed eval preparation')
        assert len(ground_images) == len(predicted_images)





