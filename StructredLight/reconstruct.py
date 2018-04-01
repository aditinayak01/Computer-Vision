# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys
from itertools import izip

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")


def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape
    color_image = cv2.imread("images/pattern001.jpg",cv2.IMREAD_COLOR)
    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))
    height,width=proj_mask.shape
      
    scan_bits = np.zeros((h, w), dtype=np.uint16)
    bgrArray  = np.zeros((height,width,3), np.uint8)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
        # mask where the pixels are ON

        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        scan_bits[on_mask]=scan_bits[on_mask] | bit_code


    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)
    color_points=[]
    camera_points = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

                # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
                # TODO: find for the camera (x,y) the projector (p_x, p_y).
                # TODO: store your points in camera_points and projector_points
                # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

            p_x,p_y=binary_codes_ids_codebook[scan_bits[y,x]]
            
            if p_x >= 1279 or p_y >= 799: # filter
                continue
            projector_points.append([[p_x,p_y]])
            camera_points.append([[x/2.0,y/2.0]])
            color_points.append([color_image[y,x]])
            bgrArray[y,x,0]=0
            bgrArray[y,x,1] = ((p_y).astype(np.float32)*255)/800
            bgrArray[y,x,2] = ((p_x).astype(np.float32)*255)/1280


    cv2.namedWindow( "Display window", cv2.WINDOW_NORMAL);
    cv2.resizeWindow('Display window', 600,600)
    cv2.imshow("Display window", bgrArray)
    cv2.waitKey(0)  
    cv2.imwrite( "correspondence.jpg", bgrArray );
    

        
    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    cam_array = np.asarray(camera_points)
    cam_array=cam_array.astype(np.float32)
    proj_array = np.asarray(projector_points).astype(np.float32)
    proj_array = proj_array.astype(np.float32)
        
    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

        # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
        # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d

        norm_cam=cv2.undistortPoints(cam_array,camera_K,camera_d)
        
        norm_proj=cv2.undistortPoints(proj_array,projector_K,projector_d)
        
        # TODO: use cv2.triangulatePoints to triangulate the normalized points
        # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
        # TODO: name the resulted 3D points as "points_3d"

        parameter_0=np.eye(4)[0:3]

        parameter_1=np.array([])
 
        parameter_1=np.concatenate((projector_R,projector_t), axis=1)


        
        points_2d=cv2.triangulatePoints(parameter_0,parameter_1,norm_cam,norm_proj)

        

        points_3d=cv2.convertPointsFromHomogeneous(points_2d.T)

        #mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
        masked_3d=[]
        filter_array=[]
        for i in range(len(points_3d)):
            if(points_3d[i][0][2]>200)& (points_3d[i][0][2] < 1400 ):
                masked_3d.append(points_3d[i])
                filter_array.append(color_points[i])

        points_3d=np.array(masked_3d)

        filter_array = np.array(filter_array)

        output_name = sys.argv[1] + "output_color.xyz"

        with open(output_name, "w") as f:
            for point,filter1 in izip(points_3d,filter_array):
                f.write("%d %d %d %d %d %d\n" % (point[0, 0], point[0, 1], point[0, 2],filter1[0, 2], filter1[0, 1], filter1[0, 0]))
        return points_3d


def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    output_name = sys.argv[1] + "output.xyz"

    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0,0], p[0,1], p[0,2]))
    #return points_3d, camera_points, projector_points

if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)

