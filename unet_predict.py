#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 2:57:34 2021

@author: Aryal007
"""
from segmentation.data.data import fetch_loaders
from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml, pathlib, pickle, warnings, torch, matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm 
from addict import Dict
import numpy as np
#from skimage.morphology import remove_small_objects, square, dilation
import pdb

def verifyDims(img, slice_hw):
    new_dim = list(img.shape)
    if img.shape[0] % slice_hw[0] !=0: new_dim[0]=(img.shape[0]//slice_hw[0] + 1)*slice_hw[0]
    if img.shape[1] % slice_hw[1] !=0: new_dim[1]=(img.shape[1]//slice_hw[1] + 1)*slice_hw[1]
    if new_dim != img.shape:
        print(f'New canvas dimensions: {new_dim}')
        temp = np.zeros(new_dim)
        temp[0:img.shape[0],0:img.shape[1]] = img
        return temp
    return img

def O(I,F,P,S):
    return (I-F+P)//S + 1

top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)
newcolors = np.vstack((top(np.linspace(0, 0.7, 128)),
                       bottom(np.linspace(0.3, 1, 128))))
OrangeBlue = ListedColormap(newcolors, name='OrangeBlue')
warnings.filterwarnings("ignore")

font = {'size'   : 20}

matplotlib.rc('font', **font)

if __name__ == "__main__":
    conf = Dict(yaml.safe_load(open('./conf/unet_predict.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    img_path = data_dir / "test/images" / conf.filename
    model_path = data_dir / "runs" / conf.run_name / "models" / "model_bestVal.pt"
    print(f'Loading Image')  
    img = mpimg.imread(img_path) #need to add verifyChannelSize()
    _img = verifyDims(img, conf["window_size"])
    #loss_fn = fn.get_loss(conf.model_opts.args.outchannels)    
    print(f'Creating Unet Instance')    
    frame = Framework(
        #loss_fn = loss_fn,
        model_opts=conf.model_opts,
        #optimizer_opts=conf.optim_opts,
        #reg_opts=conf.reg_opts,
        device=conf.device
    )
    print(f'Loading model {conf.run_name}')
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    frame.load_state_dict(state_dict)
    filename = conf.filename.split(".")[0]
    x = np.expand_dims(_img, axis=0)
    #y = np.zeros((x.shape[1], x.shape[2]))
    x = torch.from_numpy(x).float()

    print(f'Spliting Image & Predicting')
    crop = conf["window_size"][0]//4 #for stitching purposes
    row_stride=conf.window_size[0]-(2*crop)
    col_stride=conf.window_size[1]-(2*crop)
    out_height, out_width = O(_img.shape[0],conf["window_size"][0],0,row_stride), O(_img.shape[1],conf["window_size"][0],0,col_stride)
    new_height = row_stride * (out_height-1) + conf["window_size"][0]
    new_width = col_stride * (out_width-1) + conf["window_size"][1]
    y = np.zeros((new_height,new_width))
    for i in range(out_height):
        row = i*row_stride 
        for j in range(out_width):
            col = j*col_stride
            # ind = col*row + col
            current_slice = x[:, row:row+conf["window_size"][0], col:col+conf["window_size"][1], :]
            if current_slice.shape[1] != conf.window_size[0] or current_slice.shape[2] != conf.window_size[1]:
                temp = np.zeros((1, conf.window_size[0], conf.window_size[1], x.shape[3]))
                temp[:, :current_slice.shape[1], :current_slice.shape[2], :] =  current_slice
                current_slice = torch.from_numpy(temp).float()
            prediction = frame.infer(current_slice)
            prediction = np.asarray(prediction.cpu()).squeeze()[:,:,1]
            if i == 0 and j==0:
                y[row:row+conf.window_size[0],col:col+conf.window_size[1]]= prediction
            elif i == 0 and j!=0:
                y[row:row+conf.window_size[0],col+crop:col+conf.window_size[1]]= prediction[:,crop:] 
            elif i != 0 and j==0:
                y[row+crop:row+conf.window_size[0],col:col+conf.window_size[1]]= prediction[crop:,:] 
            elif i != 0 and j!=0:
                y[row+crop:row+conf.window_size[0],col+crop:col+conf.window_size[1]]= prediction[crop:,crop:] 
    
    y = y[0:img.shape[0],0:img.shape[1]]
            
    print('Saving')
    fig, plots = plt.subplots(nrows = 1, ncols=2, figsize=(20, 10))
    images = [img, y]
    titles = ["Image", "output"]

    for i, graphs in enumerate(plots.flat):
        im = graphs.imshow(images[i])
        graphs.set_title(titles[i], fontsize=20)
        graphs.axis('off')
    plt.savefig(filename+"_output_no_postprocess.png")
    plt.close(fig)

    fig, plots = plt.subplots(nrows = 1, ncols=2, figsize=(20, 10))
    '''y = (y > 0.8)
    y = remove_small_objects(y, 30000).astype(int)
    y = dilation(y, square(15)).astype(int)'''
    images = [img, y]
    titles = ["Image", "output"]
    plt.imsave(filename+"_prediction_"+conf.run_name+".png",y,cmap="gray")
    for i, graphs in enumerate(plots.flat):
        im = graphs.imshow(images[i])
        graphs.set_title(titles[i], fontsize=20)
        graphs.axis('off')
    plt.savefig(filename+"_output.png")
    plt.close(fig)
    print('Done')