# coding: utf-8
"""
Data handler for traffic sign images got from
<http://www.dis.uniroma1.it/~bloisi/ds/data/DITS-classification.zip>
"""
import math
from glob import glob
import random
import re

import numpy as np
import cv2

TRAIN_RE = re.compile(r'train\/(\d+)/')
TEST_RE = re.compile(r'test/(\d+)/')

LABELS = """animal crossing (warning)
soft verges (warning)
road narrows (warning)
bridge ahead (indication)
uneven road (warning)
no trucks (mandatory)
dead end (indication)
bend (warning)
cattle (warning)
crossroads (warning)
direction type0 (mandatory)
direction type1 (mandatory)
direction type2 (mandatory)
no stopping (mandatory)
double bend (warning)
give priority (mandatory)
traffic priority (indication)
two way traffic (warning)
national speed limits (mandatory)
generic warning type0 (warning)
generic warning type1 (warning)
give way (warning)
lane merging (warning)
junction (warning)
max height (mandatory)
no entry (mandatory)
no overtaking (mandatory)
no waiting (mandatory)
one way (indication)
parking place (indicaftion)
pedal cycles (mandatory)
zebra crossing (warning)
major road end (indication)
major road (indication)
roundabout (mandatory)
roundabout (warning)
schol crossing (warning)
sleeping policeman (warning)
slippery road (warning)
speed limit 10 (mandatory)
speed limit 30 (mandatory)
speed limit 40 (mandatory)
speed limit 50 (mandatory)
speed limit 60 (mandatory)
speed limit 60 end (mandatory)
stop (mandatory)
traffic light (warning)
no horn (mandatory)
men at work (warning)
zebra crossing (indication)
no vehicles (mandatory)
max height type0 (mandatory)
max width type0 (mandatory)
no pedestrian (mandatory)
no horse (mandatory)
no bike (mandatory)
turn (indication)
speed limit 70 (mandatory)
speed limit 90 (mandatory)""".split('\n')


class ImagesHandler():

    def __init__(self, path, width, height, color=cv2.IMREAD_GRAYSCALE,
                 crop_borders=True):
        self.nclasses = len(LABELS)
        self.path = path
        self.width = width
        self.height = height
        self.color = color
        self.crop_borders = crop_borders

        self.x_train, self.y_train = self.load_trainning_data()
        self.x_test, self.y_test = self.load_test_data()

    def process_image(self, image):
        return process_image(image, self.width, self.height, self.color, self.crop_borders)

    @staticmethod
    def to_array(images_list):
        return np.asarray(images_list)

    def _get_label(self, rexp, img_path):
        labels = np.zeros(self.nclasses, dtype=np.float32)
        label = int(rexp.findall(img_path)[0])
        labels[label] = 1.0
        return labels

    def get_shuffled_train_data(self):
        x = list(zip(self.x_train, self.y_train))
        random.shuffle(x)
        return tuple(zip(*x))

    def train_batch_iter(self, batch_size, shuffle=True):
        """
        """
        if shuffle:
            x_train, y_train = self.get_shuffled_train_data()
        else:
            x_train, y_train = self.x_train, self.y_train
        data_length = len(y_train)
        n_batches = math.ceil(data_length / batch_size)
        while True:
            for i in range(n_batches):
                first, last = i * batch_size, i * batch_size + batch_size
                yield self.to_array(x_train[first:last]), self.to_array(y_train[first:last])

    @property
    def test_data(self):
        return self.to_array(self.x_test), self.to_array(self.y_test)

    def load_trainning_data(self):
        image_files = glob('{}**/*train/**/*.png'.format(self.path), recursive=True)
        images = [self.process_image(image) for image in image_files]
        return images, [self._get_label(TRAIN_RE, img) for img in image_files]

    def load_test_data(self):
        image_files = glob('{}**/*test/**/*.png'.format(self.path), recursive=True)
        images = [self.process_image(image) for image in image_files]
        return images, [self._get_label(TEST_RE, img) for img in image_files]


def process_image(image, height, width, imread=cv2.IMREAD_GRAYSCALE, crop_borders=True):
    """
    Set the image color, resize, optimize intensity and contrast (equalizeHist)
    :param image:
    :param height:
    :param width:
    :param color_read:
    :param crop_borders:
    :return:
    """
    image = cv2.imread(image, imread)
    image = cv2.resize(image, (height, width))
    image = cv2.equalizeHist(image)
    image = cv2.equalizeHist(image)
    image = cv2.equalizeHist(image)

    if crop_borders:
        random_percentage = random.randint(3, 20)
        to_crop_height = int((random_percentage * height) / 100)
        to_crop_width = int((random_percentage * width) / 100)
        image = image[to_crop_height:height - to_crop_height, to_crop_width:width - to_crop_width]
        image = cv2.copyMakeBorder(
            image,
            top=to_crop_height,
            bottom=to_crop_height,
            left=to_crop_width,
            right=to_crop_width,
            borderType=cv2.BORDER_CONSTANT
        )
    image = image.reshape(-1)  # Pixels to a line
    # cv2.imshow('image', image)
    # cv2.waitKey(0)  # Wait until press key to destroy image
    return image


def get_label_name(label):
    return LABELS[label]
