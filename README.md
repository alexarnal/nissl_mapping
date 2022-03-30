# Nissl Mapping
## _Estimate brain regions from images of Nissl-stained rat brain tissue_

## Project Structure
```
nissl_mapping
│
└─── nissl_mapping
│   │
│   └───data
│       │   data.py
│       │   slice.py
│   │
│   └───model
│       │   frame.py
│       │   functions.py
│       │   metrics.py
│       │   unet.py
│   
└───conf
│   │   eval.yaml
│   │   predict_slices.yaml
│   │   slice_and_preprocess.yaml
│   │   unet_predict.yaml
│   │   unet_train.yaml
│   
│   .gitignore
│   README.md
│   requirements.txt
│   win_requirements.txt
│   eval.py
│   slice_and_preprocess.py
│   unet_predict.py
│   unet_train.py
```

## Structure for data directory
```
data
│
└─── test 
│   │
│   └─── images         Location to store *.PNG files.
│   └─── fx             Location to store fx labels. The png filename and its corresponding label is same.
│   └─── gpe            Location to store gpe labels. The png filename and its corresponding label is same.
│   └─── av             Location to store av labels. The png filename and its corresponding label is same.
│   └─── sch            Location to store sch labels. The png filename and its corresponding label is same.
│   └─── ma             Location to store ma labels. The png filename and its corresponding label is same.
│   └─── pvt            Location to store pvt labels. The png filename and its corresponding label is same.
│   └─── lhaai          Location to store lhaai labels. The png filename and its corresponding label is same.
│
└─── train              Location to store *.PNG files.
│   │
│   └─── images         Location to store *.PNG files.
│   └─── fx             Location to store fx labels. The png filename and its corresponding label is same.
│   └─── gpe            Location to store gpe labels. The png filename and its corresponding label is same.
│   └─── av             Location to store av labels. The png filename and its corresponding label is same.
│   └─── sch            Location to store sch labels. The png filename and its corresponding label is same.
│   └─── ma             Location to store ma labels. The png filename and its corresponding label is same.
│   └─── pvt            Location to store pvt labels. The png filename and its corresponding label is same.
│   └─── lhaai          Location to store lhaai labels. The png filename and its corresponding label is same.
│
└─── processed          Location to store train, test, val directories. Created during slice_and_preprocess.
│
└─── runs               Location to store training runs. Created during unet_train.
```
