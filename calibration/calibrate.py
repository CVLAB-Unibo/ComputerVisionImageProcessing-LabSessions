#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images
'''

import numpy as np
import cv2
import sys
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--img_folder", default="./chessboards/", type=str)
parser.add_argument("--output_folder", default="./output/", type=str)
parser.add_argument("--debug", default="./debug_calibration/", type=str)
parser.add_argument("--square_size", default=25, type=float, help="square size in mm")
parser.add_argument("--pattern_size_x", default=9, type=int, help="square inner corner on x axis")
parser.add_argument("--pattern_size_y", default=6, type=int, help="square inner corner on y axis")
args = parser.parse_args()

def splitfn(path):
    dirname = os.path.dirname(path)
    basename, ext = os.path.splitext(os.path.basename(path))
    return dirname, basename, ext

def create_3x4_RT_matrix(rvec,tvec):
    r = np.asarray(cv2.Rodrigues(rvec)[0])
    t = np.asarray(tvec)
    return np.concatenate([r,t],axis=1)

def main():
    img_names = [os.path.join(args.img_folder,f) for f in sorted(os.listdir(args.img_folder))]
    print(img_names)
    square_size = float(args.square_size)
    pattern_size = (args.pattern_size_x, args.pattern_size_y)

    debug_dir = args.debug
    output_folder = args.output_folder
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    if output_folder and not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    #Initializing inner corner 3D coordinates 
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    #print("3D coordinates of inner corners", pattern_points)

    obj_points = []
    img_points = []
    h, w = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[:2]
    print("H {}, W {}".format(h,w))

    def processImage(fn):
        print('processing {}'.format(fn))
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))

        found, corners = cv2.findChessboardCorners(img, pattern_size)
        
        if found:
            #Refining corner position to subpixel iteratively until criteria  max_count=30 or criteria_eps_error=1 is sutisfied
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
            #Image Corners 
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv2.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None

        print('           %s... OK' % fn)
        return (corners.reshape(-1, 2), pattern_points)

    # Optional multi-threading ops
    threads_num = 8
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    
    calib_file = open(os.path.join(args.output_folder,"calib.txt"),"w")
    intrisincs_str = ""
    for i in camera_matrix.reshape([-1]).tolist():
        intrisincs_str += str(i) + " "
    calib_file.write(intrisincs_str.strip() + "\n")

    dist_coefs_str = ""
    for i in dist_coefs.reshape([-1]).tolist():
        dist_coefs_str += str(i) + " "
    calib_file.write(dist_coefs_str.strip() + "\n")
    calib_file.close()

    poses_file = open(os.path.join(args.output_folder,"poses.txt"),"w")
    poses = [create_3x4_RT_matrix(rvecs[idx], tvecs[idx]) for idx, _ in enumerate(tvecs)]
    poses_strings = ""
    for pose in poses:
        extr = ""
        for i in pose.reshape([-1]).tolist():
            extr += str(i) + " "
        poses_file.write(extr.strip() + "\n")
    poses_file.close()

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    print("Rotation vectors:", rvecs)
    print("translation vectors", tvecs)

    # undistort the image with the calibration
    for fn in img_names if debug_dir else []:
        path, name, ext = splitfn(fn)
        img_found = os.path.join(debug_dir, name + '_chess.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv2.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
