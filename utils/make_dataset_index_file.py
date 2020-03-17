#!/usr/bin/env python3
import argparse
import os
from glob import glob

'''
Demo:

# Make dataset index.txt files
./dev_docker_run ./utils/make_dataset_index_file.py --input_dataset_path datasets/open_images_10_classes_200/train --output_dataset_index_path datasets/open_images_10_classes_200/train/index.txt
./dev_docker_run ./utils/make_dataset_index_file.py --input_dataset_path datasets/open_images_10_classes_200/eval --output_dataset_index_path datasets/open_images_10_classes_200/eval/index.txt


'''

IMAGE_FILE_EXTENSIONS = ['.png', '.jpeg', '.jpg', '.bmp', '.gif']


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


if __name__ == '__main__':
    # constants

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dataset_path',
        type=str,
        default=None,
        required=True,
        help='where to load classifier dataset'
    )
    parser.add_argument(
        '--input_class_names_path',
        type=str,
        default='class_names.txt',
        required=False,
        help='Where to load the class names used by the trained model.'
    )
    parser.add_argument(
        '--output_dataset_index_path',
        type=str,
        default=None,
        required=True,
        help='Where to save classifier dataset index file'
    )

    args = parser.parse_args()

    input_dataset_path = args.input_dataset_path
    if not input_dataset_path.endswith('/'):
        input_dataset_path = input_dataset_path + '/'

    class_names_txt = open(args.input_class_names_path, 'r').read()
    class_names = class_names_txt.split('\n')

    with open(args.output_dataset_index_path, 'w') as output_f:

        image_files = glob(input_dataset_path + "**/*.*")
        for file_name in image_files:
            basename = os.path.basename(file_name)
            # print(basename)

            if basename.startswith('.'):
                continue  # hidden file
            is_image = False
            for ext in IMAGE_FILE_EXTENSIONS:
                if file_name.lower().endswith(ext):
                    is_image = True
                    break

            if not is_image:
                print('not an image, skipping: {}'.format(file_name))
                continue  # skip things that arent images

            path_in_dataset = os.path.relpath(file_name, input_dataset_path)
            # print(path_in_dataset)
            parts = splitall(path_in_dataset)
            # print(parts)

            if len(parts) < 2:
                print('Image file found not in a folder, aborting: ' + file_name)
                exit(1)

            class_name = parts[0]
            #print(class_name)

            class_index = class_names.index(class_name)

            line = "{} {}\n".format(path_in_dataset, class_index)
            output_f.write(line)
