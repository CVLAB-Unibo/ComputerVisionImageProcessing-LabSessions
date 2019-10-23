import cv2
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", default="./es1/chessboard.jpg", type=str)
parser.add_argument("--intrisics_path", default="output/calib.txt", type=str)
parser.add_argument("--output_folder", default="./output_es1/", type=str)
parser.add_argument("--square_size", default=26.5, type=float, help="square size in mm")
parser.add_argument("--pattern_size_x", default=8, type=int, help="square inner corner on x axis")
parser.add_argument("--pattern_size_y", default=5, type=int, help="square inner corner on y axis")
args = parser.parse_args()

img_name = args.img_path

square_size = float(args.square_size)
pattern_size = (args.pattern_size_x, args.pattern_size_y)

output_folder = args.output_folder
if output_folder and not os.path.isdir(output_folder):
    os.mkdir(output_folder)

#Initializing inner corner 3D coordinates 
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

def decode_intrisics(line):
    intr = np.fromstring(line.strip(), sep=" ")
    intr = intr.reshape([3,3])
    return intr

def decode_lensdist(line):
    dist = np.fromstring(line.strip(), sep=" ")
    return dist

def create_4x4_RT_matrix(rvec,tvec):
    r = np.asarray(cv2.Rodrigues(rvec)[0])
    t = np.asarray(tvec)
    mat3x4 = np.concatenate([r,t],axis=1)
    mat4x4 = np.concatenate([mat3x4, np.array([[0,0,0,1]])], axis=0)
    return mat4x4

def processImage(fn):
    print('processing {}'.format(fn))
    img = cv2.imread(fn, 0)
    h, w = img.shape[:2]

    if img is None:
        print("Failed to load", fn)
        return None
    print("H {}, W {}".format(h,w))

    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))

    found, corners = cv2.findChessboardCorners(img, pattern_size)
    
    if found:
        #Refining corner position to subpixel iteratively until criteria  max_count=30 or criteria_eps_error=1 is sutisfied
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
        #Image Corners 
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if not found:
        print('chessboard not found')
        return None

    print('           %s... OK' % fn)
    return (corners.reshape(-1, 2), pattern_points)

calib = open(args.intrisics_path).readlines()
camera_matrix = decode_intrisics(calib[0])
lens_dist = decode_lensdist(calib[1])
corners, pattern_points = processImage(img_name)

homography = cv2.findHomography(pattern_points, corners)[0] #from 3D to 2D

_, rvec, tvec = cv2.solvePnP(pattern_points, corners, camera_matrix, lens_dist)
rt = create_4x4_RT_matrix(rvec, tvec)

def es1a_extr_intr():
    # Es 1 a
    corner_grid_coords_3D = (4,2)
    point3D_homogeneous = np.asarray([[args.square_size *corner_grid_coords_3D[0] ],[args.square_size* corner_grid_coords_3D[1]], [0], [1]])
    point2D_homogeneous = np.matmul(camera_matrix, np.matmul(rt,point3D_homogeneous)[:3])
    point2D = point2D_homogeneous/point2D_homogeneous[-1,0]

    print("3D: ", point3D_homogeneous.reshape([-1])[:-1])
    print("2D: ", np.round(point2D.reshape([-1])[:-1]))

    corner_draw = cv2.imread(img_name)
    cv2.circle(corner_draw, (point2D[0], point2D[1]), 100, (0,0,255), 10)
    plt.imshow(corner_draw)
    plt.show()
    cv2.imwrite(os.path.join(args.output_folder, "corner.png"), corner_draw)

def es1a():
    # Es 1 a
    corner_grid_coords_3D = (4,2)
    point3D_homogeneous = np.asarray([[args.square_size *corner_grid_coords_3D[0] ],[args.square_size* corner_grid_coords_3D[1]], [1]])
    point2D_homogeneous = np.matmul(homography, point3D_homogeneous)
    point2D = point2D_homogeneous/point2D_homogeneous[-1,0]

    print("3D: ", point3D_homogeneous.reshape([-1])[:-1])
    print("2D: ", np.round(point2D.reshape([-1])[:-1]))

    corner_draw = cv2.imread(img_name)
    cv2.circle(corner_draw, (point2D[0], point2D[1]), 100, (0,0,255), 10)
    plt.imshow(corner_draw)
    plt.show()
    cv2.imwrite(os.path.join(args.output_folder, "corner.png"), corner_draw)

def es1b_extr_intr():
    # Es 1 b
    point2D_homogeneous = (1579, 2168, 1)

    ppm = np.matmul(camera_matrix, rt[:3,:])
    homography = np.concatenate([ppm[:, :2], ppm[:,3:]], axis=1)

    point3D_homogeneous = np.matmul(np.linalg.inv(homography), point2D_homogeneous)
    point3D = point3D_homogeneous/point3D_homogeneous[-1]
    point3D[-1] = 0
    print("2D: ", point2D_homogeneous[:-1])
    print("3D: ", point3D)

def es1b():
    # Es 1 b
    point2D_homogeneous = (1579, 2168, 1)

    point3D_homogeneous = np.matmul(np.linalg.inv(homography), point2D_homogeneous)
    point3D = point3D_homogeneous/point3D_homogeneous[-1]
    point3D[-1] = 0

    print("2D: ", point2D_homogeneous[:-1])
    print("3D: ", point3D)

def es2_extr_intr():
    image = cv2.imread(img_name)
    ppm = np.matmul(camera_matrix, rt[:3,:])
    homography = np.concatenate([ppm[:, :2], ppm[:,3:]], axis=1)
    
    point2D_homogeneous_a = (2722, 1160, 1)
    point2D_homogeneous_b = (2686, 2675, 1)

    point3D_homogeneous_a = np.matmul(np.linalg.inv(homography), point2D_homogeneous_a)
    point3D_a = point3D_homogeneous_a/point3D_homogeneous_a[-1]
    point3D_a[-1] = 0

    point3D_homogeneous_b = np.matmul(np.linalg.inv(homography), point2D_homogeneous_b)
    point3D_b = point3D_homogeneous_b/point3D_homogeneous_b[-1]
    point3D_b[-1] = 0

    print("2D: ", point2D_homogeneous_a[:-1], point2D_homogeneous_b[:-1])
    print("3D: ", point3D_a, point3D_b)

    cv2.circle(image, (point2D_homogeneous_a[0], point2D_homogeneous_a[1]), 100, (0,0,255), 10)
    cv2.circle(image, (point2D_homogeneous_b[0], point2D_homogeneous_b[1]), 100, (0,0,255), 10)
    
    dist = np.linalg.norm(point3D_a-point3D_b)
    print("{:.2f}mm".format(dist))

    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.show()

def es2():
    image = cv2.imread(img_name)

    point2D_homogeneous_a = (2722, 1160, 1)
    point2D_homogeneous_b = (2686, 2675, 1)

    point3D_homogeneous_a = np.matmul(np.linalg.inv(homography), point2D_homogeneous_a)
    point3D_a = point3D_homogeneous_a/point3D_homogeneous_a[-1]
    point3D_a[-1] = 0

    point3D_homogeneous_b = np.matmul(np.linalg.inv(homography), point2D_homogeneous_b)
    point3D_b = point3D_homogeneous_b/point3D_homogeneous_b[-1]
    point3D_b[-1] = 0

    print("2D: ", point2D_homogeneous_a[:-1], point2D_homogeneous_b[:-1])
    print("3D: ", point3D_a, point3D_b)

    cv2.circle(image, (point2D_homogeneous_a[0], point2D_homogeneous_a[1]), 100, (0,0,255), 10)
    cv2.circle(image, (point2D_homogeneous_b[0], point2D_homogeneous_b[1]), 100, (0,0,255), 10)
    
    dist = np.linalg.norm(point3D_a-point3D_b)
    print("{:.2f}mm".format(dist))

    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.show()

def es3():
    image = cv2.imread(args.img_path)
    image2project = cv2.imread("es3/stregatto.jpg") 

    h, w = image.shape[0], image.shape[1]

    maxWidth = image2project.shape[1]
    maxHeight = image2project.shape[0]

    rect = np.array([
        [2537, 519],
        [2573, 3480],
        [530, 3670],
        [488, 380]
    ], dtype = "float32")
    
    # print(mapping.shape)
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(dst,rect)
    
    white = np.ones([maxHeight,maxWidth,3],dtype=np.uint8)*255
    warp_mask = cv2.warpPerspective(white, M, (w, h))
    warp_mask = np.equal(warp_mask, np.array([0,0,0]))
    
    warped = cv2.warpPerspective(image2project, M, (w, h))

    warped[warp_mask] = image[warp_mask]
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.show()

def es4():
    def four_point_transform(image, pts):
        def order_points(pts):
            # initialzie a list of coordinates that will be ordered
            # such that the first entry in the list is the top-left,
            # the second entry is the top-right, the third is the
            # bottom-right, and the fourth is the bottom-left
            rect = np.zeros((4, 2), dtype = "float32")
        
            # the top-left point will have the smallest sum, whereas
            # the bottom-right point will have the largest sum
            s = pts.sum(axis = 1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
        
            # now, compute the difference between the points, the
            # top-right point will have the smallest difference,
            # whereas the bottom-left will have the largest difference
            diff = np.diff(pts, axis = 1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
        
            # return the ordered coordinates
            return rect

        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
    
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
    
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
    
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order        
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
    
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
        # return the warped image
        return warped

    image = cv2.imread(args.img_path)
    points = np.array([[93,3823], [124,160], [2448,472], [2404, 3572]])
    warped = four_point_transform(image, points)

    plt.imshow(warped)
    plt.show()