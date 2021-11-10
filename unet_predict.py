#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 2:57:34 2021

@author: Aryal007
"""
from segmentation.data.data import fetch_loaders
from segmentation.model.frame import Framework
from segmentation.data.slice import add_index
import segmentation.model.functions as fn

import yaml, pathlib, pickle, warnings, torch, matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm 
from addict import Dict
import numpy as np
import pdb

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
    img_path = data_dir / "test_full" / conf.filename
    model_path = data_dir / "runs" / conf.run_name / "models" / "model_final.pt"
    img = mpimg.imread(img_path)
    loss_fn = fn.get_loss(conf.model_opts.args.outchannels)    
        
    frame = Framework(
        loss_fn = loss_fn,
        model_opts=conf.model_opts,
        optimizer_opts=conf.optim_opts,
        reg_opts=conf.reg_opts
    )

    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    frame.load_state_dict(state_dict)
    filename = conf.filename.split(".")[0]
    x = np.expand_dims(img, axis=0)
    y = np.zeros((x.shape[1], x.shape[2]))
    x = torch.from_numpy(x).float()
    
    for row in range(0, x.shape[1], conf.window_size[0]):
        for column in range(0, x.shape[2], conf.window_size[1]):
            current_slice = x[:, row:row+conf["window_size"][0], column:column+conf["window_size"][1], :]
            if current_slice.shape[1] != conf.window_size[0] or current_slice.shape[2] != conf.window_size[1]:
                temp = np.zeros((1, conf.window_size[0], conf.window_size[1], x.shape[3]))
                temp[:, :current_slice.shape[1], :current_slice.shape[2], :] =  current_slice
                current_slice = torch.from_numpy(temp).float()
            #mask = current_slice.squeeze()[:,:,:3].sum(dim=2) == 0
            prediction = frame.infer(current_slice)
            prediction = torch.nn.Softmax(3)(prediction)
            prediction = np.asarray(prediction.cpu()).squeeze()[:,:,1]
            #prediction[mask] = 0
            prediction = (prediction > 0.3).astype(int)
            endrow_dest = row+conf.window_size[0]
            endrow_source = conf.window_size[0]
            endcolumn_dest = column+conf.window_size[0]
            endcolumn_source = conf.window_size[1]
            if endrow_dest > y.shape[0]:
                endrow_source = y.shape[0] - row
                endrow_dest = y.shape[0]
            if endcolumn_dest > y.shape[1]:
                endcolumn_source = y.shape[1] - column
                endcolumn_dest = y.shape[1]
            try:
                y[row:endrow_dest, column:endcolumn_dest] = prediction[0:endrow_source, 0:endcolumn_source]
            except Exception as e:
                print("Something wrong with indexing!")
            
    fig, plots = plt.subplots(nrows = 1, ncols=2, figsize=(20, 10))
    images = [img, y]
    titles = ["Image", "output"]

    for i, graphs in enumerate(plots.flat):
        im = graphs.imshow(images[i])
        graphs.set_title(titles[i], fontsize=20)
        graphs.axis('off')
    plt.savefig(filename+"_output.png")
    plt.close(fig)