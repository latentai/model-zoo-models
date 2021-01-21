#  Copyright (c) 2020 by LatentAI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "LatentAI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.

import argparse
import os


def save_config(base_dir, args):
    # [print(vals) for vals in vars(args)]
    path = base_dir + '/' + args.name_of_experiment + '/'
    with open(path + 'config.txt', 'w') as f:
        for vals in vars(args):
            f.write(str(vals) + ' ' + str(getattr(args, vals)))
            f.write('\n')
    f.close()


def config(name_of_experiment='lenet_on_mnist',
           batch_size=32,
           num_epochs=5,
           learning_rate=.0002,
           max_number_bb_per_gt=10,
           image_size=(28, 28),
           num_channels=1,
           num_classes=10,
           lambda_bit_loss=1e-5,
           lambda_distillation_loss=0.01,
           lambda_regularization=0.0001,
           weight_decay=.0002,
           learning_rate_decay=None):
    
    if image_size is None or (not isinstance(image_size, tuple)):
        raise ValueError('pass correct image size of the data')

    parser = argparse.ArgumentParser()
    name_of_experiment = name_of_experiment + '_' + 'adam' + '_weight_decay_' + str(
        weight_decay) + '_lam_bl_' + str(lambda_bit_loss) + '_lam_dl_' + str(
        lambda_distillation_loss)

    parser.add_argument(
        '--name_of_experiment',
        type=str,
        default=name_of_experiment,
        help='Name of the experiment')
    parser.add_argument(
        '--batch_size', type=int, default=batch_size, help='batch size')
    parser.add_argument(
        '--num_epochs', type=int, default=num_epochs, help='num of epochs')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=learning_rate,
        help='learning rate')
    parser.add_argument(
        '--learning_rate_decay',
        type=float,
        default=0.0,
        help='decay for the learning rate')
    parser.add_argument(
        '--image_size',
        type=tuple,
        default=(image_size[0], image_size[1], num_channels),
        help='image size')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=num_classes,
        help='number of classes in dataset')
    parser.add_argument(
        '--lambda_bit_loss',
        type=float,
        default=lambda_bit_loss,
        help='lambda for bit loss')
    parser.add_argument(
        '--lambda_distillation_loss',
        type=float,
        default=lambda_distillation_loss,
        help='lambda for distillation loss')
    parser.add_argument(
        '--lambda_regularization',
        type=float,
        default=0.0001,
        help='lambda of the regularization')
    #gtc related cli parameters
    parser.add_argument("--basedirectory", default = "",
                        help="base directory for the output model")
    parser.add_argument("--print_layers_bits", default=True, type=bool, help="print bits per layer?")
    args = parser.parse_args()
    base_dir = args.basedirectory
    path_expt = base_dir + '/' + args.name_of_experiment + '/'
    model_path = path_expt + 'model/'
    logs_path = path_expt + 'logs/'
    if not os.path.exists(path_expt):
        os.makedirs(path_expt)
        os.makedirs(model_path)
        os.makedirs(logs_path)

    save_config(base_dir, args)
    return args
