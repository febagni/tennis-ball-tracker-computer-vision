import argparse
import queue
import pandas as pd 
import pickle
import imutils
import os
from PIL import Image, ImageDraw
import cv2 
import numpy as np
import torch
import sys
import time

from sktime.datatypes._panel._convert import from_2d_array_to_nested
from court_detector import CourtDetector
from Models.tracknet import trackNet
from utils import get_video_properties, get_dtype
from detection import *
from pickle import load


# parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--full_trajectory", type=int, default=0)

args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path
full_trajectory = args.full_trajectory

n_classes = 256
save_weights_path = 'WeightsTracknet/model.1'

if output_video_path == "":
    # output video in same path
    output_video_path = "VideoOutput/" + input_video_path.split('.')[0] + "_video_output.mp4"

# get videos properties
video = cv2.VideoCapture(input_video_path)
fps, length, v_width, v_height = get_video_properties(video)

# start from first frame
currentFrame = 0

# width and height in TrackNet
width, height = 640, 360 # input size of TrackNet
img, img1, img2 = None, None, None

trajectory_n = 16
if full_trajectory==1:
   trajectory_n = length - 1
else:
   trajectory_n = 16
#   trajectory_n = length - 1
   

# load TrackNet model
modelFN = trackNet
m = modelFN(n_classes, input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights(save_weights_path)

# In order to draw the trajectory of tennis, we need to save the coordinate of previous n frames
q = queue.deque()
for i in range(0, trajectory_n):
    q.appendleft(None)

# save prediction images as videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (v_width, v_height))

# court
court_detector = CourtDetector()

coords = []
frame_i = 0
frames = []
t = []

while True:
  ret, frame = video.read()
  frame_i += 1

  if ret:
    if frame_i == 1:
      print('Detecting the court...')
      lines = court_detector.detect(frame)
    else: # then track it
      lines = court_detector.track_court(frame)
    
    for i in range(0, len(lines), 4):
      x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
      cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 5)
    new_frame = cv2.resize(frame, (v_width, v_height))
    frames.append(new_frame)
  else:
    break
video.release()
print('Finished!')

# ACABOU A PARTE DA QUADRA E COMECA DA BOLA

video = cv2.VideoCapture(input_video_path)
frame_i = 0

last = time.time() # start counting 
# while (True):
for img in frames:
    print(f"BALL TRACKING: frame {currentFrame} out of {length} frames.")
    frame_i += 1

    # detect the ball
    # img is the frame that TrackNet will predict the position
    # since we need to change the size and type of img, copy it to output_img
    output_img = img

    # resize it
    img = cv2.resize(img, (width, height)) # for tracknet input size
    # input must be float type
    img = img.astype(np.float32)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    X = np.rollaxis(img, 2, 0)
    # predict heatmap
    pr = m.predict(np.array([X]))[0]

    # since TrackNet output is ( net_output_height*model_output_width , n_classes )
    # so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2)

    # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
    pr = pr.astype(np.uint8)

    # reshape the image size as original input image
    heatmap = cv2.resize(pr, (v_width, v_height))

    # heatmap is converted into a binary image by threshold method.
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # find the circle in image with 2<=radius<=7 (empirical values?)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                              maxRadius=7)

    
    #related to PIL library 
    PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    # check if there have any tennis balls to be detected
    if circles is not None:
        if len(circles) == 1:
            x, y = map(int, circles[0][0][:2])
            coords.append([x, y])
        else:
            coords.append(None)
    else:
        coords.append(None)

    t.append(time.time() - last)
    q.appendleft(coords[-1])
    q.pop()

    # draw current frame prediction and previous n frames as yellow circle, total: n+1 frames
    draw = ImageDraw.Draw(PIL_image)
    trajectory_points = []

    for i in range(trajectory_n):
        if q[i] is not None:
            draw_x, draw_y = q[i]

            if full_trajectory == 0:
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw.ellipse(bbox, outline='yellow')
            else:
                # if draw_x<1400 or draw_y< 900:
                    # print(f"X: {draw_x} | Y: {draw_y}")
                trajectory_points.append((draw_x, draw_y))
    
    if full_trajectory == 1 and trajectory_points:
        outlier_threshold = 100
        filtered_points = [trajectory_points[0]]
        for j in range(1, len(trajectory_points)):
            distance = ((trajectory_points[j][0] - trajectory_points[j-1][0])**2 + (trajectory_points[j][1] - trajectory_points[j-1][1])**2) ** 0.5
            # print(f"DISTANCE: {distance}")
            if distance < outlier_threshold:
                filtered_points.append(trajectory_points[j])
        draw.line(trajectory_points, fill='yellow', width=2)
        
    del draw


    # Convert PIL image format back to opencv image format
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

    output_video.write(opencvImage)

    # next frame
    currentFrame += 1

# everything is done, release the video
video.release()
output_video.release()
