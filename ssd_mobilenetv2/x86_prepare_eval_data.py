import os
import tvm
from tvm.contrib import graph_runtime
import json
import yaml
import pickle
from tqdm import tqdm
from glob import glob
from ssd_utils import BBoxUtility
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from xml.etree import ElementTree
from tensorflow.keras.preprocessing import image
from x86model import Model
print('Segmentation fault (core dumped)')

base = "./"
image_dir = base + "imgs/"
name_dir = base + "resources/data/imagenet/"
#
# Load the test image, match the expected input shape of 224,224,3
#
# img_name = image_dir + "000030.jpg"

# dtype = "float32"
# img = Image.open(img_name).resize((224, 224))
# data = np.array(img)[np.newaxis, :].astype(dtype)
# global_data = preprocess_input(data[:, :, :, ::-1], mode="tf")
input_name = "input_1"
# shape_dict = {input_name: global_data.shape}
ctx = tvm.cpu(0)
base = "baseline_compiled_tvm_int/bin/"
# loaded_graph = open(base + "modelDescription.json").read()
# loaded_lib = tvm.runtime.load_module(base + "modelLibrary.so")
# loaded_params = bytearray(open(base + "modelParams.params", "rb").read())
#
# graphjson = json.loads(loaded_graph)
# if 'leip' in list(graphjson.keys()):
#     del graphjson['leip']
#     loaded_graph = json.dumps(graphjson)

# m = graph_runtime.create(loaded_graph, loaded_lib, ctx)
# m.set_input(input_name, tvm.nd.array(global_data))
# m.load_params(loaded_params)

priors = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       'priorFiles/prior_boxes_ssd300MobileNetV2_224_224.pkl'), 'rb'))

np.set_printoptions(suppress=True)
NUM_CLASSES = 21
THRESHOLD = 0.6
CLASSES = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
           'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
           'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
           'Sheep', 'Sofa', 'Train', 'Tvmonitor']
bbox_util = BBoxUtility(NUM_CLASSES, priors)


class X86_Model():
    def __init__(self, args):
        with open(args.path_to_settings, 'r') as fp:
            sets = yaml.safe_load(fp)
        self.sets = sets
        # loaded_graph = open(base + "modelDescription.json").read()
        # loaded_lib = tvm.runtime.load_module(base + "modelLibrary.so")
        # self.loaded_params = bytearray(open(base + "modelParams.params", "rb").read())
        #
        # graphjson = json.loads(loaded_graph)
        # if 'leip' in list(graphjson.keys()):
        #     del graphjson['leip']
        #     loaded_graph = json.dumps(graphjson)
        #
        # graph = loaded_graph
        # lib = loaded_lib
        # m = graph_runtime.create(graph, lib, ctx)
        # print('input_name: {}'.format(input_name))
        # m.load_params(self.loaded_params)
        # self.m = m
        self.m = Model()
        self.m.load(base)


    def inference(self, _data):
        print('img: {}'.format(_data.shape))
        self.m.set_input(input_name, tvm.nd.array(_data))
        tvm_output = self.m.get_output(0).asnumpy()
        # self.m.set_input(input_name, tvm.nd.array(global_data))
        # tvm_output = self.m.get_output(0)
        # tvm_output = tvm_output.asnumpy()

        print('detection_out: {}'.format(bbox_util.detection_out(tvm_output)[0]))

        return tvm_output

    def create_model_prediction(self, n_images=400):
        priors = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               'priorFiles/prior_boxes_ssd300MobileNetV2_224_224.pkl'), 'rb'))

        np.set_printoptions(suppress=True)

        bbox_util = BBoxUtility(NUM_CLASSES, priors)

        inputs = []
        images = []
        result_images = []
        annotation_files = []

        print('Prepare : {} files for evaluation. '.format(n_images))
        input_shape = (self.sets['img_height'], self.sets['img_width'], 3)

        with open(os.path.join(self.sets['dataset_dir'], 'VOC2007/ImageSets/Main/test.txt'), 'r') as annot_f:
            for annotation in tqdm(list(annot_f)[:n_images]):
                try:
                    img_path = os.path.join(self.sets['dataset_dir'], 'VOC2007/JPEGImages/') + annotation.split(' ')[0].strip() + '.jpg'
                    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
                    img = image.img_to_array(img)
                    result_images.append(img_path)
                    images.append(img)
                    inputs.append(img.copy())
                    annotation_files.append(annotation)
                except Exception as e:
                    print('Error while opening file.', e)

        result_detections = []

        # inputs = preprocess_input(np.array(inputs)[:, :, :, ::-1], mode="tf")
        inputs = np.array(inputs)
        inputs = preprocess_input(inputs)

        print('inputs: {}'.format(inputs.shape))

        results = []
        for img in tqdm(inputs):
            # self.m._model.set_input(input_name, tvm.nd.array(img))
            # self.m._model.run()
            tvm_output = self.m.predict_on_batch(img)
            # ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=1)
            # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond

            tvm_output = self.m._model.get_output(0)
            img_result = bbox_util.detection_out(tvm_output.asnumpy())
            results.append(img_result)

        results = np.array(results)
        results = np.squeeze(results, axis=1)
        print('results: {}'.format(results.shape))

        for i, img in tqdm(enumerate(images)):
            det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]

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
                    ['{:.2f}'.format(xmin), '{:.2f}'.format(ymin), '{:.2f}'.format(xmax), '{:.2f}'.format(ymax),
                     label_name,
                     '{:.2f}'.format(score)])
            result_detections.append(detections)

        print('Test images: {}'.format(len(result_images)))
        print('result_detections: {}'.format(len(result_detections)))

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

        GROUND_TRUTH_LABELS = os.path.join(self.sets['dataset_dir'], 'VOC2007/Annotations')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_settings', help='Path to dataset')
    parser.add_argument('--pretrained_weights', help='Model pretrained weights')

    args = parser.parse_args()
    x86_model = X86_Model(args)

    x86_model.create_model_prediction()
