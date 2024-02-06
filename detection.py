import operator
import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from scipy import signal
import imutils

from court_detector import CourtDetector
from sort import Sort
from utils import get_video_properties, get_dtype
import matplotlib.pyplot as plt


class DetectionModel:
    def __init__(self, dtype=torch.FloatTensor):
        # STILL GOTTA REFACTOR THE ATTRIBUTES
        self.detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detection_model.type(dtype)  # Also moves model to GPU if available
        self.detection_model.eval()
        self.dtype = dtype
        self.PERSON_LABEL = 1
        self.RACKET_LABEL = 43
        self.BALL_LABEL = 37
        self.PERSON_SCORE_MIN = 0.85
        self.PERSON_SECONDARY_SCORE = 0.3
        self.RACKET_SCORE_MIN = 0.6
        self.BALL_SCORE_MIN = 0.6
        self.v_width = 0
        self.v_height = 0
        self.player_1_boxes = []
        self.player_2_boxes = []
        self.persons_boxes = {}
        self.persons_dists = {}
        self.persons_first_appearance = {}
        self.counter = 0
        self.num_of_misses = 0
        self.last_frame = None
        self.current_frame = None
        self.next_frame = None
        self.movement_threshold = 200
        self.mot_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.05)

def draw_ball_position(frame, court_detector, xy, i):
        """
        Calculate the ball position of both players using the inverse transformation of the court and the x, y positions
        """
        inv_mats = court_detector.game_warp_matrix[i]
        coord = xy
        img = frame.copy()
        # Ball locations
        if coord is not None:
          p = np.array(coord,dtype='float64')
          ball_pos = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
          transformed = cv2.perspectiveTransform(ball_pos, inv_mats)[0][0].astype('int64')
          cv2.circle(frame, (transformed[0], transformed[1]), 35, (0,255,255), -1)
        else:
          pass
        return img 


# def create_top_view(court_detector, detection_model, xy, fps):
#     """
#     Creates top view video of the gameplay
#     """
#     coords = xy[:]
#     court = court_detector.court_reference.court.copy()
#     court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
#     v_width, v_height = court.shape[::-1]
#     court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
#     out = cv2.VideoWriter('VideoOutput/minimap.mp4',cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (v_width, v_height))
#     # players location on court
#     smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)
#     i = 0 
#     for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
#         frame = court.copy()
#         frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 45, (255, 0, 0), -1)
#         if feet_pos_2[0] is not None:
#             frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 45, (255, 0, 0), -1)
#         draw_ball_position(frame, court_detector, coords[i], i)
#         i += 1
#         out.write(frame)
#     out.release()


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolation(coords):
  coords =coords.copy()
  x, y = [x[0] if x is not None else np.nan for x in coords], [x[1] if x is not None else np.nan for x in coords]

  xxx = np.array(x) # x coords
  yyy = np.array(y) # y coords

  nons, yy = nan_helper(xxx)
  xxx[nons]= np.interp(yy(nons), yy(~nons), xxx[~nons])
  nans, xx = nan_helper(yyy)
  yyy[nans]= np.interp(xx(nans), xx(~nans), yyy[~nans])

  newCoords = [*zip(xxx,yyy)]

  return newCoords

def diff_xy(coords):
  coords = coords.copy()
  diff_list = []
  for i in range(0, len(coords)-1):
    if coords[i] is not None and coords[i+1] is not None: # if ball detected in both frames, do the distance
      point1 = coords[i]
      point2 = coords[i+1]
      diff = [abs(point2[0] - point1[0]), abs(point2[1] - point1[1])]
      diff_list.append(diff)
    else:
      diff_list.append(None) # if ball not detected in both frames, add None to the list
  
  xx, yy = np.array([x[0] if x is not None else np.nan for x in diff_list]), np.array([x[1] if x is not None else np.nan for x in diff_list])
  
  return xx, yy # diff in x and diff in y

def remove_outliers(x, y, coords):
  ids = set(np.where(x > 50)[0]) & set(np.where(y > 50)[0])
  for id in ids:
    left, middle, right = coords[id-1], coords[id], coords[id+1]
    if left is None:
      left = [0]
    if  right is None:
      right = [0]
    if middle is None:
      middle = [0]
    MAX = max(map(list, (left, middle, right)))
    if MAX == [0]:
      pass
    else:
      try:
        coords[coords.index(tuple(MAX))] = None
      except ValueError:
        coords[coords.index(MAX)] = None
