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

from court_detector import CourtDetector
from Models.tracknet import trackNet
from utils import *
from detection import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_video_path", type=str, default="", help="Path to save the output video file.")
    parser.add_argument("--full_trajectory", type=int, default=0, help="Flag to determine if full trajectory should be drawn.")
    return parser.parse_args()


def initialize_tracknet(n_classes, height, width, save_weights_path):
    model = trackNet(n_classes, input_height=height, input_width=width)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.load_weights(save_weights_path)
    return model


def setup_output_video(input_video_path, output_video_path, fps, v_width, v_height):
    if not output_video_path:
        output_video_path = f"VideoOutput/{os.path.splitext(input_video_path)[0]}_video_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (v_width, v_height))
    return output_video


def detect_and_rectify_court(video, v_width, v_height):
    court_detector = CourtDetector()
    frame_i = 0
    frames = []
    src_points = None  # Source points for perspective transformation
    dst_points = np.float32([[0, 0], [v_width, 0], [v_width, v_height], [0, v_height]])  # Destination points for top-down view

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_i += 1
        if frame_i == 1:
            print('Detecting the court...')
            lines = court_detector.detect(frame)
            # Define source points for perspective transformation
            left_top = (lines[0], lines[1])
            right_top = (lines[2], lines[3])
            left_bottom = (lines[4], lines[5])
            right_bottom = (lines[6], lines[7])
            src_points = np.float32([
                left_bottom, right_bottom,  # Bottom-left, bottom-right
                right_top, left_top # Top-right, top-left
            ])
        else:
            lines = court_detector.track_court(frame)
        
        # Draw the court lines
        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 5)
        
        # Apply perspective transformation to get the top-down view
        if src_points is not None:
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            top_down_view = cv2.warpPerspective(frame, M, (v_width, v_height))
            frames.append(top_down_view)
        else:
            new_frame = cv2.resize(frame, (v_width, v_height))
            frames.append(new_frame)
    
    print('Court detection and rectification finished!')
    return frames


def detect_court(video, v_width, v_height):
    court_detector = CourtDetector()
    frame_i = 0
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_i += 1
        if frame_i == 1:
            print('Detecting the court...')
            lines = court_detector.detect(frame)
        else:
            lines = court_detector.track_court(frame)
        
        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 5)
        
        left_top = (lines[0], lines[1])
        right_top = (lines[2], lines[3])
        left_bottom = (lines[4], lines[5])
        right_bottom = (lines[6], lines[7])
            
            # Draw the corners on the frame
        corners = [left_top, right_top, left_bottom, right_bottom]
        for idx, point in enumerate(corners):
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(idx + 1), (int(point[0]), int(point[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        new_frame = cv2.resize(frame, (v_width, v_height))
        frames.append(new_frame)
    
    print('Court detection finished!')
    return frames


def detect_and_rectify_court_2(video, v_width, v_height):
    court_detector = CourtDetector()
    frame_i = 0
    frames = []
    src_points = None  # Source points for perspective transformation
    dst_points = np.float32([[0, 0], [v_width, 0], [v_width, v_height], [0, v_height]])  # Destination points for top-down view

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_i += 1
        if frame_i == 1:
            print('Detecting the court...')
            lines = court_detector.detect(frame)
            # Define source points for perspective transformation
            left_top = (lines[0], lines[1])
            right_top = (lines[2], lines[3])
            left_bottom = (lines[4], lines[5])
            right_bottom = (lines[6], lines[7])
            src_points = np.float32([
                left_bottom, right_bottom,  # Bottom-left, bottom-right
                right_top, left_top  # Top-right, top-left
            ])

            # Extend the lines to ensure the entire court is within the rectified view
            extend_ratio = 1.8  # Extend the lines by 20%
            extended_src_points = extend_court_bounds(src_points, extend_ratio)
        else:
            lines = court_detector.track_court(frame)

        # Draw the court lines
        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 5)
        
        # Apply perspective transformation to get the top-down view
        if src_points is not None:
            M = cv2.getPerspectiveTransform(extended_src_points, dst_points)
            top_down_view = cv2.warpPerspective(frame, M, (v_width, v_height))
            frames.append(top_down_view)
        else:
            new_frame = cv2.resize(frame, (v_width, v_height))
            frames.append(new_frame)
    
    print('Court detection and rectification finished!')
    return frames


def extend_court_bounds(points, ratio):
    """ Extend the boundaries of the court to ensure it fits within the rectified image """
    center = np.mean(points, axis=0)
    extended_points = center + (points - center) * ratio
    return extended_points


def compute_vanishing_point(p1, p2):
    line = np.cross(p1, p2)
    return line / line[2]

def compute_homography(src_points):
    # Convert points to homogeneous coordinates
    pts_h = np.array([[x, y, 1] for x, y in src_points])
    
    # Compute lines
    line1 = np.cross(pts_h[0], pts_h[1])  # line between top-left and top-right
    line2 = np.cross(pts_h[2], pts_h[3])  # line between bottom-right and bottom-left
    line3 = np.cross(pts_h[1], pts_h[2])  # line between top-right and bottom-right
    line4 = np.cross(pts_h[3], pts_h[0])  # line between bottom-left and top-left

    # Compute vanishing points
    vp1 = np.cross(line1, line2)
    vp2 = np.cross(line3, line4)

    # Normalize vanishing points
    vp1 = vp1 / vp1[2]
    vp2 = vp2 / vp2[2]

    # Compute line at infinity
    line_at_infinity = np.cross(vp1, vp2)
    line_at_infinity = line_at_infinity / line_at_infinity[2]

    # Compute affine rectification matrix
    H_affine = np.array([[1, 0, 0], 
                         [0, 1, 0], 
                         [line_at_infinity[0], line_at_infinity[1], 1]])
    
    return H_affine

def detect_and_rectify_court_3(video, v_width, v_height):
    court_detector = CourtDetector()
    frame_i = 0
    frames = []
    src_points = None  # Source points for affine transformation
    dst_points = np.float32([[0, 0], [v_width, 0], [v_width, v_height], [0, v_height]])  # Destination points for top-down view

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_i += 1
        if frame_i == 1:
            print('Detecting the court...')
            lines = court_detector.detect(frame)
            # Define source points for affine transformation
            left_top = (lines[0], lines[1])
            right_top = (lines[2], lines[3])
            left_bottom = (lines[4], lines[5])
            right_bottom = (lines[6], lines[7])
            src_points = np.float32([
                left_top, right_top,  # Top-left, top-right
                right_bottom, left_bottom  # Bottom-right, bottom-left
            ])
        else:
            lines = court_detector.track_court(frame)
        
        # Draw the court lines
        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
        
        # Apply affine transformation to get the top-down view
        if src_points is not None:
            H_affine = compute_homography(src_points)
            rectified_frame = cv2.warpPerspective(frame, H_affine, (v_width, v_height))
            frames.append(rectified_frame)
        else:
            new_frame = cv2.resize(frame, (v_width, v_height))
            frames.append(new_frame)
    
    print('Court detection and rectification finished!')
    return frames

def preprocess_image(img, width, height):
    output_img = img.copy()
    img = cv2.resize(img, (width, height)).astype(np.float32)
    X = np.rollaxis(img, 2, 0)
    return output_img, X

 
def predict_heatmap(model, X, height, width, n_classes):
    pr = model.predict(np.array([X]))[0]
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2).astype(np.uint8)
    return pr


def update_coords(circles, coords):
    if circles is not None and len(circles) == 1:
        x, y = map(int, circles[0][0][:2])
        coords.append([x, y])
    else:
        coords.append(None)
    return coords


def create_heatmap(pr, v_width, v_height):
    heatmap = cv2.resize(pr, (v_width, v_height))
    _, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
    return heatmap


def detect_circles(heatmap):
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
                               param1=50, param2=2, minRadius=2, maxRadius=7)
    return circles


def draw_trajectory(PIL_image, q, trajectory_n, full_trajectory):
    draw = ImageDraw.Draw(PIL_image)
    trajectory_points = []

    for i in range(trajectory_n):
        if q[i] is not None:
            draw_x, draw_y = q[i]
            if full_trajectory == 0:
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw.ellipse(bbox, outline='yellow')
            else:
                trajectory_points.append((draw_x, draw_y))

    if full_trajectory == 1 and trajectory_points:
        draw.line(trajectory_points, fill='yellow', width=2)

    del draw


def process_frame(img, model, currentFrame, length, width, height, v_width, v_height, n_classes, coords, q, t, trajectory_n, full_trajectory, output_video, last):
    print(f"BALL TRACKING: frame {currentFrame} out of {length} frames.")

    output_img, X = preprocess_image(img, width, height)
    pr = predict_heatmap(model, X, height, width, n_classes)
    heatmap = create_heatmap(pr, v_width, v_height)
    circles = detect_circles(heatmap)

    PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    coords = update_coords(circles, coords)

    t.append(time.time() - last)
    q.appendleft(coords[-1])
    q.pop()

    draw_trajectory(PIL_image, q, trajectory_n, full_trajectory)

    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
    output_video.write(opencvImage)

    currentFrame += 1
    return currentFrame, coords, q, t


def main():
    args = parse_arguments()

    input_video_path = args.input_video_path
    output_video_path = args.output_video_path
    full_trajectory = args.full_trajectory

    n_classes = 256
    save_weights_path = 'WeightsTracknet/modelo.h5'

    video = cv2.VideoCapture(input_video_path)
    fps, length, v_width, v_height = get_video_properties(video)

    width, height = 640, 360
    trajectory_n = length - 1 if full_trajectory == 1 else 16

    model = initialize_tracknet(n_classes, height, width, save_weights_path)

    q = queue.deque([None] * trajectory_n)

    output_video = setup_output_video(input_video_path, output_video_path, fps, v_width, v_height)

    frames = detect_and_rectify_court_3(video, v_width, v_height)
    video.release()

    video = cv2.VideoCapture(input_video_path)
    currentFrame = 0
    coords = []
    t = []
    last = time.time()

    for img in frames:
        currentFrame, coords, q, t = process_frame(
            img, model, currentFrame, length, width, height, v_width, v_height, 
            n_classes, coords, q, t, trajectory_n, full_trajectory, output_video, last
        )

    video.release()
    output_video.release()


if __name__ == "__main__":
    main()
