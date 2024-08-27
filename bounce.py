import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import cv2
import pickle

def compute_rectified_coords(M, coords_x, coords_y, peaks, print_coords=False):
    # Convert the coordinates to homogeneous coordinates
    X = coords_x[peaks]
    Y = coords_y[peaks]

    rectified_coords = []
    for x,y in zip(X,Y):
        OG_coords = np.array([x, y, 1])
        rectified_coord = M @ OG_coords
        rectified_x = int(rectified_coord[0] / rectified_coord[2])
        rectified_y = int(rectified_coord[1] / rectified_coord[2])
        if print_coords:
            print(f"Original coordinates: ({x}, {y}), Rectified coordinates: {rectified_x}, {rectified_y}")

        rectified_coords.append((rectified_x, rectified_y))

    return np.array(rectified_coords)
