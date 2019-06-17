# COCO Dataset - 2014 Training, Validation, and Test datasets for Captioning Task

## Overview
This dataset is from Microsoft's [COCO](http://cocodataset.org/#home) dataset. It uses the training, validation, and test sets for the captioning task.

## How to use
First, run `pipenv run install_cocoapi` to download and install COCO API used to extract images captions. 

Then, the `COCO2014` object can be instantiated, as follows:

```python
coco2014_dataset = COCO2014()
```

Note the above instantiation will create COCO 2014 training set. To create validation and test sets, set either `is_validation_set` or `is_test_set` to `True`.

As soon as there is an instance, the initializer will perform the download, extraction, and mapping of images to captions. All of the files will be stored under `data/raw/coco2014`.
