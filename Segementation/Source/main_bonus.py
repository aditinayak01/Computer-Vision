
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
import sys
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=18.470)
    
    segments_ids = np.unique(segments)
    
    
    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)
BLUE = [255,0,0]         # sure BG
RED = [0,0,255]   # sure FG
global value
value=BLUE
drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def line_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode,value
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        cv2.circle(im,(former_x,former_y),3, value, -1)
        cv2.circle(img,(former_x,former_y),3, value, -1)



    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
            		cv2.circle(im,(former_x,former_y),3, value, -1)
                	cv2.circle(img,(former_x,former_y),3, value, -1)
                	                        

   
                
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.circle(im,(former_x,former_y),3, value, -1)
            cv2.circle(img,(former_x,former_y),3, value, -1)

 

im = cv2.imread(sys.argv[1],cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
im1=im.copy()
cv2.imshow('output',im1)
height, width, channels = im.shape
img = np.zeros([height, width, 3], dtype=np.uint8)
img.fill(255)
cv2.namedWindow("main image")
cv2.setMouseCallback('main image',line_draw)

centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(im1)

while(1):
    cv2.imshow('masking',img)
    cv2.moveWindow('masking',im.shape[1]+10,90)
    cv2.imshow('main image',im)
    

    k=cv2.waitKey(1)&0xFF
    if k==ord('2'):

        break
    elif k == ord('0'): # BG drawing
	print(" mark background regions with left mouse button \n")
        value = BLUE
    elif k == ord('1'): # FG drawing
	print(" mark foreground regions with left mouse button \n")
        value = RED
    elif k == ord('n'): # FG drawing
	fg_segments, bg_segments = find_superpixels_under_marking(img, superpixels)
	fgbg_superpixels= np.array((fg_segments, bg_segments))
        fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
        bg_cumulative_hist=cumulative_histogram_for_superpixels(bg_segments, color_hists)
        norm_hists = normalize_histograms(color_hists)
        fgbg_hists= np.array((fg_cumulative_hist, bg_cumulative_hist))
        graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)
        segmask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
        segmask = np.uint8(segmask * 255)
        cv2.imshow('output',segmask)
	

    elif k==27:
	break





cv2.destroyAllWindows()