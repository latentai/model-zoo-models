#!/usr/bin/env python3

import dload

dload.save_unzip("https://model-zoo-data.latentai.io/open_images_10_classes_200_train/2020-03-17-00-45-41/c38f244b60271296dc68c5a9d3f83537.zip", "./datasets/")
dload.save_unzip("https://model-zoo-data.latentai.io/open_images_10_classes_200_eval/2020-03-17-00-57-38/38511464608f326cc33a5076dd06f658.zip", "./datasets/")

print('Downloaded!')
