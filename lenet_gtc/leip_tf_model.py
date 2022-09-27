"""
This script contains example of LeNet model consitsting of 5 layers,
prepared with LEIP framework.
"""

import tensorflow.compat.v1 as tf
from leip.compress.training.gtc.activation import Activation as gtcActivation
from leip.compress.training.gtc.conv2d import Conv2d as gtcConv2d
from leip.compress.training.gtc.dense import Dense as gtcDense
from leip.compress.training.gtc.flatten import Flatten as gtcFlatten
from leip.compress.training.gtc.reshape import Reshape as gtcReshape
from leip.compress.training.gtc.pooling import MaxPooling2D as gtcMaxPooling2D
from leip.compress.training.gtcModel import GTCModel
from leip.compress.training.quantization import GTCQuantization
from leip.compress.training.quantization import IdentityQuantization


# create LeNet model using keras Sequential
def L1(args, x_placeholder):
    L1Model = GTCModel()
    L1Model.add(x_placeholder)
    L1Model.add(
        gtcConv2d(
            quantizer=GTCQuantization(),
            filters=6,
            kernel_size=5,
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.initializers.glorot_normal(),
            input_shape=args.image_size,
            use_bias=False))
    L1Model.add(
        gtcActivation(
            quantizer=IdentityQuantization(),
            activation='relu',
            trainable=False))
    L1Model.add(
        gtcMaxPooling2D(
            quantizer=IdentityQuantization(),
            strides=2,
            pool_size=2))
    L1Model.add(
        gtcConv2d(
            quantizer=GTCQuantization(),
            filters=10,
            kernel_size=5,
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.initializers.glorot_normal(),
            use_bias=False))
    L1Model.add(
        gtcActivation(quantizer=IdentityQuantization(), activation='relu'))
    L1Model.add(
        gtcMaxPooling2D(
            quantizer=IdentityQuantization(),
            strides=2,
            pool_size=2))
    L1Model.add(gtcFlatten())
    L1Model.add(gtcReshape((49,10,1)))
    # L1Model.add(
    #     gtcDense(
    #         quantizer=GTCQuantization(),
    #         units=128,
    #         kernel_initializer=tf.initializers.glorot_normal(),
    #         bias_initializer=tf.initializers.glorot_normal(),
    #         use_bias=False))
    L1Model.add(
        gtcConv2d(
            quantizer=GTCQuantization(),
            filters=128,
            kernel_size=(49,10), # the size of the kernel
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.initializers.glorot_normal(),
            padding="VALID",
            use_bias=False))
    L1Model.add(gtcActivation(quantizer=GTCQuantization(), activation='relu'))
    # L1Model.add(
    #     gtcDense(
    #         quantizer=GTCQuantization(),
    #         units=128,
    #         kernel_initializer=tf.initializers.glorot_normal(),
    #         bias_initializer=tf.initializers.glorot_normal(),
    #         use_bias=False))
    L1Model.add(gtcReshape((128,1,1)))
    L1Model.add(
        gtcConv2d(
            quantizer=GTCQuantization(),
            filters=128,
            kernel_size=(128,1), # the size of the kernel
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.initializers.glorot_normal(),
            padding="VALID",
            use_bias=False))
    L1Model.add(
        gtcActivation(quantizer=IdentityQuantization(), activation='relu'))
    # L1Model.add(
    #     gtcDense(
    #         quantizer=GTCQuantization(),
    #         units=args.num_classes,
    #         kernel_initializer=tf.initializers.glorot_normal(),
    #         bias_initializer=tf.initializers.glorot_normal(),
    #         use_bias=False))
    L1Model.add(gtcReshape((128,1,1)))
    L1Model.add(
        gtcConv2d(
            quantizer=GTCQuantization(),
            filters=args.num_classes,
            kernel_size=(128,1), # the size of the kernel
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.initializers.glorot_normal(),
            padding="VALID",
            use_bias=False))
    L1Model.add(gtcFlatten())
    return L1Model