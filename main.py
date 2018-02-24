#!/usr/bin/env python
'''
This is a re-implementation of the following paper:
"Attention-based Deep Multiple Instance Learning"
I got very similar results but some data augmentation techniques not used here
https://128.84.21.199/pdf/1802.04712.pdf
*---- Jiawen Yao--------------*
'''


import numpy as np
import time
from utl import Cell_Net
from random import shuffle
import argparse
from keras.models import Model
from utl.dataset import load_dataset
from utl.data_aug_op import random_flip_img, random_rotate_img
import glob
import scipy.misc as sci
import tensorflow as tf

from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import matplotlib.pyplot as plt

import os

def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=1e-4, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=0.0005, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epoch to train',
                        default=100, type=int)
    parser.add_argument('--useGated', dest='useGated',
                        help='use Gated Attention',
                        default=False, type=int)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def generate_batch(path):
    bags = []
    for each_path in path:
        name_img = []
        img = []
        img_path = glob.glob(each_path + '/*.bmp')
        num_ins = len(img_path)

        label = int(each_path.split('/')[-2])

        if label == 1:
            curr_label = np.ones(num_ins,dtype=np.uint8)
        else:
            curr_label = np.zeros(num_ins, dtype=np.uint8)
        for each_img in img_path:
            img_data = np.asarray(sci.imread(each_img), dtype=np.float32)
            #img_data -= 255
            img_data[:, :, 0] -= 123.68
            img_data[:, :, 1] -= 116.779
            img_data[:, :, 2] -= 103.939
            img_data /= 255
            # sci.imshow(img_data)
            img.append(np.expand_dims(img_data,0))
            name_img.append(each_img.split('/')[-1])
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, curr_label, name_img))

    return bags


def Get_train_valid_Path(Train_set, train_percentage=0.8):
    """
    Get path from training set
    :param Train_set:
    :param train_percentage:
    :return:
    """
    import random
    indexes = np.arange(len(Train_set))
    random.shuffle(indexes)

    num_train = int(train_percentage*len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val


def test_eval(model, test_set):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on testing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """
    num_test_batch = len(test_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)
    for ibatch, batch in enumerate(test_set):
        result = model.test_on_batch(x=batch[0], y=batch[1])
        test_loss[ibatch] = result[0]
        test_acc[ibatch] = result[1]
    return np.mean(test_loss), np.mean(test_acc)

def train_eval(model, train_set, irun, ifold):
    """Evaluate on training set. Use Keras fit_generator
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    Returns
    -----------------
    model_name: saved lowest val_loss model's name
    """
    batch_size = 1
    model_train_set, model_val_set = Get_train_valid_Path(train_set, train_percentage=0.9)

    from utl.DataGenerator import DataGenerator
    train_gen = DataGenerator(batch_size=1, shuffle=True).generate(model_train_set)
    val_gen = DataGenerator(batch_size=1, shuffle=False).generate(model_val_set)

    model_name = "Saved_model/" + "_Batch_size_" + str(batch_size) + "epoch_" + "best.hd5"

    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1)

    EarlyStop = EarlyStopping(monitor='val_loss', patience=20)

    callbacks = [checkpoint_fixed_name, EarlyStop]

    history = model.fit_generator(generator=train_gen, steps_per_epoch=len(model_train_set)//batch_size,
                                             nb_epoch=args.max_epoch, validation_data=val_gen,
                                            validation_steps=len(model_val_set)//batch_size, callbacks=callbacks)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['bag_accuracy']
    val_acc = history.history['val_bag_accuracy']

    fig = plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = 'Results/' + str(irun) + '_' + str(ifold) + "_loss_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)


    fig = plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = 'Results/' + str(irun) + '_' + str(ifold) + "_val_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)

    return model_name


def model_training(input_dim, dataset, irun, ifold):

    train_bags = dataset['train']
    test_bags = dataset['test']

    # convert bag to batch
    train_set = generate_batch(train_bags)
    test_set = generate_batch(test_bags)

    model = Cell_Net.cell_net(input_dim, args, useMulGpu=False)

    # train model
    t1 = time.time()
    num_batch = len(train_set)
    # for epoch in range(args.max_epoch):
    model_name = train_eval(model, train_set, irun, ifold)

    print("load saved model weights")
    model.load_weights(model_name)

    test_loss, test_acc = test_eval(model, test_set)

    t2 = time.time()
    #

    print 'run time:', (t2 - t1) / 60.0, 'min'
    print 'test_acc={:.3f}'.format(test_acc)

    return test_acc



if __name__ == "__main__":

    args = parse_args()

    print 'Called with args:'
    print args

    input_dim = (27,27,3)

    run = 1
    n_folds = 10
    acc = np.zeros((run, n_folds), dtype=float)
    data_path = 'dataset/Colon/Patches'

    for irun in range(run):
        dataset = load_dataset(dataset_path=data_path, n_folds=n_folds, rand_state=irun)
        for ifold in range(n_folds):
            print 'run=', irun, '  fold=', ifold
            acc[irun][ifold] = model_training(input_dim, dataset[ifold], irun, ifold)
    print 'mi-net mean accuracy = ', np.mean(acc)
    print 'std = ', np.std(acc)

