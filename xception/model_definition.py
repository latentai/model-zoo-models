from keras.applications.imagenet_utils import preprocess_input

image_size = (299, 299)

def preprocess_imagenet(img):
    return preprocess_input(img, mode='tf')
