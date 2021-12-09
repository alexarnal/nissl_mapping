#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:26:56 2021

@author: mibook
"""
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon, box
from rasterio.features import rasterize
from shapely.ops import cascaded_union
import matplotlib.image as mpimg
import numpy as np
import os, shutil
import pdb

def read_label(filename):
    img = mpimg.imread(filename)
    return img

def read_img(filename):
    img = mpimg.imread(filename)
    #x = np.mean(img, axis=2)
    x = img
    return x

def save_slices(filename, img, label, split, **conf):
    def verify_slice_size(slice, conf):
        if slice.shape[0] != conf["window_size"][0] or slice.shape[1] != conf["window_size"][1]:
            temp = np.zeros((conf["window_size"][0], conf["window_size"][1], slice.shape[2]))
            temp[0:slice.shape[0], 0:slice.shape[1],:] = slice
            slice = temp
        return slice

    def filter_percentage(img, label, percentage):
        labelled_pixels = np.sum(label)
        total_pixels = label.shape[0] * label.shape[1]
        if labelled_pixels/total_pixels < percentage:
            randnum = np.random.rand()
            threshold = 0.999
            if np.std(img) <= 0.06 and np.mean(img) >= 0.8:
                threshold = 0.8
            if randnum <= threshold:
                return False
        return True

    def save_slice(arr, filename):
        np.save(filename, arr)

    if not os.path.exists(conf["out_dir"]+split):
        os.makedirs(conf["out_dir"]+split)

    slicenum = 0
    for row in range(0, img.shape[0], conf["window_size"][0]-conf["overlap"]):
        for column in range(0, img.shape[0], conf["window_size"][1]-conf["overlap"]):
            label_slice = label[row:row+conf["window_size"][0], column:column+conf["window_size"][1]]
            label_slice = verify_slice_size(label_slice, conf)
            img_slice = img[row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
            img_slice = verify_slice_size(img_slice, conf)

            if filter_percentage(img_slice, label_slice, conf["filter"]):
                save_slice(label_slice, conf["out_dir"]+split+"/label_"+str(filename)+"_slice_"+str(slicenum))
                save_slice(img_slice, conf["out_dir"]+split+"/img_"+str(filename)+"_slice_"+str(slicenum))
                print(f"Saved image {filename} slice {slicenum}")
            slicenum += 1

def remove_and_create(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

def train_test_shuffle(out_dir, train_split, val_split, test_split):
    train_path = out_dir + "train/"
    remove_and_create(train_path)
    val_path = out_dir + "val/"
    remove_and_create(val_path)
    test_path = out_dir + "test/"
    remove_and_create(test_path)

    slices = [x for x in os.listdir(out_dir) if (x.endswith('.npy') and "tiff" in x )]
    n_tiffs = len(slices)
    random_index = np.random.permutation(n_tiffs)
    savepath = train_path
    for count, index in enumerate(random_index):
        if count > int(n_tiffs*train_split):
            savepath = val_path
        if count > int(n_tiffs*(train_split+val_split)):
            savepath = test_path
        tiff_filename = slices[index]
        mask_filename = tiff_filename.replace("tiff","mask")
        shutil.move(out_dir+tiff_filename, savepath+tiff_filename)
        shutil.move(out_dir+mask_filename, savepath+mask_filename)