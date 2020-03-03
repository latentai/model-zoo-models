import os

import numpy as np
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras import backend as K

# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet50.ResNet50()

#tf.keras.experimental.export_saved_model(model, 'resnet50_keras')

sess = K.get_session()

saver = tf.compat.v1.train.Saver()
os.makedirs('resnet50_keras_checkpoint', exist_ok=True)
saver.save(sess, "resnet50_keras_checkpoint/ckpt")

saver.restore(sess, "model_save/new_model")

#model.save('resnet50_keras', save_format='tf')


# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img("path_to_image.jpg", target_size=(224, 224))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = resnet50.decode_predictions(predictions, top=9)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))
