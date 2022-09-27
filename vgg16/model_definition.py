from keras.applications.imagenet_utils import preprocess_input

image_size = (224, 224)

def preprocess_imagenet_caffe(img):
    return preprocess_input(img, mode='caffe')
