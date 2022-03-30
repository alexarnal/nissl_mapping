#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:27:56 2021

@author: mibook
""" 
import os, yaml
import numpy as np
from addict import Dict
import segmentation.data.slice as fn
import warnings
warnings.filterwarnings("ignore")
import skimage.measure
import pdb

np.random.seed(923)

conf = Dict(yaml.safe_load(open('./conf/slice_and_preprocess.yaml')))

label_filenames = [x for x in os.listdir(conf.labels_dir)]
img_filenames = label_filenames
fn.remove_and_create(conf.out_dir)
val_filenames = ['lvl23.png','lvl30.png','lvl21_1.png', 'lvl32.png'] #val_filenames = ['lvl31.png','lvl22.png','lvl24.png']
train_filenames = ['lvl31.png','lvl28.png','lvl22.png','lvl24.png'] #train_filenames = ['lvl32.png','lvl30.png','lvl28.png','lvl23.png','lvl21_1.png']

for i, (label_filename, img_filename) in enumerate(zip(label_filenames, img_filenames)):
    split = ""
    if label_filename in val_filenames:
        split = "val"
    elif label_filename in train_filenames:
        split = "train"
    if split == "train" or split == "val":
        try:
            print(conf.labels_dir+label_filename)
            label = fn.read_label(conf.labels_dir+label_filename)
            print(label.shape, np.max(label))
        except:
            continue
        ##label = np.mean(label, axis=2)
        ##label = (label==1).astype(int)
        #label = skimage.measure.block_reduce(label, (2,2), np.max)
        ##label = np.expand_dims(label, axis=2)
        img = fn.read_img(conf.image_dir+img_filename)
        #img = skimage.measure.block_reduce(img, (2,2), np.max)
        #img = np.expand_dims(img, axis=2)
        #fn.save_slices(i, img, label, split, **conf)
        fn.save_slices_v2(i, img, label, split, **conf)

test_label_filenames = [x for x in os.listdir(conf.test_labels_dir)]
test_img_filenames = test_label_filenames
test_filenames = ['lvl21.png','lvl25.png']

for i, (label_filename, img_filename) in enumerate(zip(test_label_filenames, test_img_filenames)):
    split = ""
    if label_filename in test_filenames:
        split = "test"
    if split == "test":
        try:
            print(conf.test_labels_dir+label_filename)
            label = fn.read_label(conf.test_labels_dir+label_filename)
            print(label.shape, np.max(label))
        except:
            continue
        img = fn.read_img(conf.test_image_dir+img_filename)
        #img = fn.addPositionChannel(img)
        print (label.shape)
        print (img.shape) 
        #img = skimage.measure.block_reduce(img, (2,2), np.max)
        #img = np.expand_dims(img, axis=2)
        fn.save_test_slices(label_filename.split('.')[0], img, label, split, **conf)

print("Saving slices completed!!!")