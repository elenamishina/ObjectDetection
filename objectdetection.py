#!/usr/bin/env python
# coding: utf-8


#import torchvision
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from ipywidgets import IntProgress
#import torchvision.transforms as transforms


import numpy
import numpy as np
import cv2
from PIL import Image
import time
#import json
#import pydantic

import torchvision
import torchvision.transforms as transforms

import tempfile
import streamlit as st

CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def main():
    print("Welcome Object Detection demo")
    streamlit()



def process(model, transform, frame):
    image_tf = transform(frame)
    start_time = time.time()
    predictions = model([image_tf])
    print("--- %s sec ---" % (time.time() - start_time))

    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(predictions[0]['boxes'].detach().numpy())]
    pred_score = list(predictions[0]['scores'].detach().numpy())
    pred_class = [CLASSES[i] for i in list(predictions[0]['labels'].numpy())]

    img_result = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

    for i in range(len(pred_boxes)):
        score = pred_score[i]
        if(pred_class[i]=='person'):
            if(score > 0.7):
                bbox = pred_boxes[i]
                start_point = (int(bbox[0]), int(bbox[1]))
                end_point = (int(bbox[2]), int(bbox[3]))
                color = (255, 0, 0)
                thickness = 2
                cv2.rectangle(img_result, start_point, end_point, color, thickness)

    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    return img_result


def load_image(image_file):
    img = Image.open(image_file)
    return img


def streamlit():
    st.header("Object Detection demo")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])

    image_file = st.file_uploader("Choose a image file to process",type=['png','jpeg'])
    if image_file is not None:
        img = load_image(image_file)
        #load_model(img)
        img_result = process(model, transform, img)
        st.image(np.asarray(img_result), width=None)

    st.markdown('#')

    video_file = st.file_uploader("Choose a video file to process",type=['mp4','avi'])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        #bytes_data = video_file.read()
        #st.video(bytes_data)

        cap = cv2.VideoCapture(tfile.name)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        #frame_width = int(cap.get(3))
        #frame_height = int(cap.get(4))
        #print(frame_width, frame_height)
        
        stframe = st.empty()
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_result = process(model, transform, frame)
            stframe.image(img_result)

        cap.release()



if __name__ == "__main__":
        main()



