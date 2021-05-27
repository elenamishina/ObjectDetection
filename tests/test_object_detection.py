#!/usr/bin/env python
# coding: utf-8


#import torchvision
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from ipywidgets import IntProgress
#import torchvision.transforms as transforms


import numpy
import numpy as np
from PIL import Image
import time
import os
import json
from jsondiff import diff

import torchvision
import torchvision.transforms as transforms

dir_name = os.path.abspath(os.path.dirname(__file__))
test_image_path = os.path.join(dir_name, "persons.jpeg")
test_result_path = os.path.join(dir_name, "persons.json")

test_image = Image.open(test_image_path).convert("RGB")
with open(test_result_path, "r") as j_file:
    test_result = json.loads(j_file.read())

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
transform = transforms.Compose([transforms.ToTensor()])


def _to_json_dict_with_strings(dictionary):
    if type(dictionary) != dict:
        return str(dictionary)
    d = {k: _to_json_dict_with_strings(v) for k, v in dictionary.items()}
    return d

def to_json(dic):
    import types
    import argparse

    if type(dic) is dict:
        dic = dict(dic)
    else:
        dic = dic.__dict__
    return _to_json_dict_with_strings(dic)

def test_process_object_detection():
    image_tf = transform(test_image)
    predictions = model([image_tf])
    print("ok")
    df = diff(to_json(predictions[0]), test_result)
    print(df)
    assert to_json(predictions[0]).items()==test_result.items()



