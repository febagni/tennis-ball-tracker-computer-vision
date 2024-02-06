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
parser.add_argument("--minimap", type=int, default=0)
parser.add_argument("--bounce", type=int, default=0)

args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path
minimap = args.minimap
bounce = args.bounce

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

# load TrackNet model
modelFN = trackNet
m = modelFN(n_classes, input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights(save_weights_path)

# In order to draw the trajectory of tennis, we need to save the coordinate of previous n frames
q = queue.deque()
for i in range(0, 16):
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
    for i in range(0, 16):
        if q[i] is not None:
            draw_x = q[i][0]
            draw_y = q[i][1]
            bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
            draw = ImageDraw.Draw(PIL_image)
            draw.ellipse(bbox, outline='yellow')
            del draw

    # Convert PIL image format back to opencv image format
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

    output_video.write(opencvImage)

    # next frame
    currentFrame += 1

# everything is done, release the video
video.release()
output_video.release()


# A PARTIR DAQUI - ACABOU A PARTE DA BOLA E COMECA A PARTE DE PREDICAO DE BOUNCE

for _ in range(3):
  x, y = diff_xy(coords) # coords sao as coordenadas da bola durante todos os frames 
  # x e y vao ser a array da diferenca entre as coordenadas da bola em cada frame, para as situacoes em q a bola foi detectada
  # em dois frames seguidos  
  remove_outliers(x, y, coords)

# interpolation
coords = interpolation(coords)

# velocty 
Vx = []
Vy = []
V = []
frames = [*range(len(coords))]

for i in range(len(coords)-1):
  p1 = coords[i] # coords is the array of the ball's coordinates
  p2 = coords[i+1]
  t1 = t[i] # time array defined way earlier 
  t2 = t[i+1]
  x = (p1[0]-p2[0])/(t1-t2)
  y = (p1[1]-p2[1])/(t1-t2)
  Vx.append(x)
  Vy.append(y)

for i in range(len(Vx)):
  vx = Vx[i]
  vy = Vy[i]
  v = (vx**2+vy**2)**0.5 #total velocity magnitude would be the square of both
  V.append(v)

xy = coords[:]

if bounce == 1:
  # Predicting Bounces - makes a df with the points the coordinates and the velocity at that frame
  test_df = pd.DataFrame({'x': [coord[0] for coord in xy[:-1]], 'y':[coord[1] for coord in xy[:-1]], 'V': V})

  print(test_df)
  # df.shift
  for i in range(20, 0, -1): 
    test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
  for i in range(20, 0, -1): 
    test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
  for i in range(20, 0, -1): 
    test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)

  test_df.drop(['x', 'y', 'V'], 1, inplace=True)

  Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
        'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1']]
  Xs = from_2d_array_to_nested(Xs.to_numpy())

  Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
        'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
        'lagY_3', 'lagY_2', 'lagY_1']]
  Ys = from_2d_array_to_nested(Ys.to_numpy())

  Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
        'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
        'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
        'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
  Vs = from_2d_array_to_nested(Vs.to_numpy())

  X = pd.concat([Xs, Ys, Vs], 1)

  print(X)

  # load the pre-trained classifier  
  clf = load(open('clf.pkl', 'rb'))

  predcted = clf.predict(X)
  idx = list(np.where(predcted == 1)[0])
  idx = np.array(idx) - 10
  
  if minimap == 1:
    video = cv2.VideoCapture('VideoOutput/video_with_map.mp4')
  else:
    video = cv2.VideoCapture(output_video_path)

  output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(video.get(cv2.CAP_PROP_FPS))
  length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  fourcc = cv2.VideoWriter_fourcc(*'XVID')

  print(fps)
  print(length)

  output_video = cv2.VideoWriter('VideoOutput/final_video.mp4', fourcc, fps, (output_width, output_height))
  i = 0
  while True:
    ret, frame = video.read()
    if ret:
      # if coords[i] is not None:
      if i in idx:
        center_coordinates = int(xy[i][0]), int(xy[i][1])
        radius = 3
        color = (255, 0, 0)
        thickness = -1
        cv2.circle(frame, center_coordinates, 10, color, thickness)
      i += 1
      output_video.write(frame)
    else:
      break

  video.release()
  output_video.release()
