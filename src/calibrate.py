import cv2
import numpy as np
import glob
import os

def calibrate():
    # Setup checkerboard (6x8 internal vertices)
    pattern = (6, 8)
    square_size = 25 # mm
    
    # Object points: (0,0,0), (25,0,0) ... 
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size

    obj_points = [] 
    img_points = [] 

    images = glob.glob('../calibration/*.png')
    if not images:
        print("Error: No images found in ../calibration/")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            print(f"Corners found in {fname}")

    # Calibrate
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (960, 540), None, None)
    
    # Save the Intrinsic Matrix K
    np.savez('camera_params.npz', K=K, dist=dist)
    print("\nIntrinsic Matrix (K) saved:\n", K)
    return K, dist

if __name__ == "__main__":
    calibrate()