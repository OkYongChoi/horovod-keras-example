import argparse
import tensorflow as tf
import random as rn
import numpy as np
np.random.seed(965)
rn.seed(965)
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, \
                                    Add, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D
import os
from time import time

from tensorflow_addons.optimizers import NovoGrad
# import Horovod
import horovod.tensorflow.keras as hvd
# initialize Horovod
hvd.init()
# pin to a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

parser = argparse.ArgumentParser(description='CIFAR-10 Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--wd', type=float, default=0.000005,
                    help='weight decay')
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=float, default=2,
                    help='Number of epochs that meet target before stopping')

args = parser.parse_args()

# Define the function that creates the model

def cbr(x, conv_size):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(conv_size, (3,3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x

def conv_block(x, conv_size, scale_input = False):
    x_0 = x
    if scale_input:
        x_0 = Conv2D(conv_size, (1, 1), activation='linear', padding='same')(x_0)

    x = cbr(x, conv_size)
    x = Dropout(0.01)(x)
    x = cbr(x, conv_size)
    x = Add()([x_0, x])

    return x

def create_model():

    # Implementation of WideResNet (depth = 16, width = 10) based on keras_contrib
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/wide_resnet.py

    inputs = Input(shape=(32, 32, 3))

    x = cbr(inputs, 16)

    x = conv_block(x, 160, True)
    x = conv_block(x, 160)
    x = MaxPooling2D((2, 2))(x)
    x = conv_block(x, 320, True)
    x = conv_block(x, 320)
    x = MaxPooling2D((2, 2))(x)
    x = conv_block(x, 640, True)
    x = conv_block(x, 640)
    x = GlobalAveragePooling2D()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)

    opt = NovoGrad(lr=args.base_lr, grad_averaging=True)

    # Wrap the optimizer in a Horovod distributed optimizer
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

# Only set `verbose` to `1` if this is the root worker.
# Otherwise, it should be zero.
if hvd.rank() == 0:
    verbose = 1
else:
    verbose = 0

num_classes = 10
data_augmentation = True

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions
img_rows, img_cols = 32, 32
num_classes = 10

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = create_model()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Callbacks. 

class PrintTotalTime(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = round(time() - self.start_time, 2)
        print(f"Elapsed training time through epoch {epoch+1}: {elapsed_time}")

    def on_train_end(self, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Total training time: {total_time}") 

class StopAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target=0.85, patience=2, verbose=0):
        self.target = target
        self.patience = patience
        self.verbose = verbose
        self.stopped_epoch = 0
        self.met_target = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > self.target:
            self.met_target += 1
        else:
            self.met_target = 0

        if self.met_target >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose == 1:
            print(f'Early stopping after epoch {self.stopped_epoch + 1}')

# Define a function for a simple learning rate decay over time
def lr_schedule(epoch):
    
    if epoch < 15:
        return args.base_lr
    if epoch < 25:
        return 1e-1 * args.base_lr
    if epoch < 35:
        return 1e-2 * args.base_lr
    return 1e-3 * args.base_lr

callbacks = [
    StopAtAccuracy(target=args.target_accuracy, patience=args.patience, verbose=verbose),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule)
]

# broadcast initial variable states from the first worker to
# all others by adding the broadcast global variables callback.
callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

# average the metrics among workers at the end of every epoch
# by adding the metric average callback.
callbacks.append(hvd.callbacks.MetricAverageCallback())

if verbose:
    callbacks.append(PrintTotalTime())

# This will do preprocessing and realtime data augmentation:
datagen = image.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size),
        callbacks=callbacks,
        epochs=args.epochs,
        verbose=verbose,
        # avoid shuffling for reproducible training
        shuffle=False,
        steps_per_epoch=int(len(y_train)/(args.batch_size)) // hvd.size(),
        validation_data=(x_test, y_test),
        workers=4)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=verbose)
if verbose:
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
