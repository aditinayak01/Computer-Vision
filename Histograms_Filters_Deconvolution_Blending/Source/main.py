# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2 
import numpy as np
from matplotlib import pyplot as plt

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):
    img_out = img_in                                                  #Take the image as input
    b, g, r = cv2.split(img_out)                                      #Split the image on 3 channels  namely blue red and green and apply histogram equalization on every channel
    hb,bins= np.histogram(b.flatten(),256,[0,256])                    #Calculate the histogram of Blue channel
    cdf = hb.cumsum()                                                 #Calculate the cdf of histogram of Blue Channel
    cdf_m1 = np.ma.masked_equal(cdf,0)                                #Masking off all cdf values which are 0
    cdf_m1 = (cdf_m1 - cdf_m1.min())*255/(cdf_m1.max()-cdf_m1.min())  #Calculating the new masked cdf according the equation.and getting the mask for generating normalised histogram
    cdf = np.ma.filled(cdf_m1,0).astype('uint8')                      #Filling the cdf array with 0 if none entries found else returing cdf array abd typecasting it to integer type
    B = cdf[b]                                                        #Assigning the equalised pixel values to single channeled B array


    #Repeating the Steps mentioned above for Red and Green Channels respectively and the getting the single channeled arrays such R and G

    hr,bins= np.histogram(r.flatten(),256,[0,256])                   
    cdf = hr.cumsum()                               		     
    cdf_m2 = np.ma.masked_equal(cdf,0)                                  
    cdf_m2 = (cdf_m2 - cdf_m2.min())*255/(cdf_m2.max()-cdf_m2.min())
    cdf = np.ma.filled(cdf_m2,0).astype('uint8')
    R = cdf[r]
                                   
    hg,bins=np.histogram(g.flatten(),256,[0,256])                    
    cdf = hg.cumsum()
    cdf_m3 = np.ma.masked_equal(cdf,0)
    cdf_m3 = (cdf_m3 - cdf_m3.min())*255/(cdf_m3.max()-cdf_m3.min())
    cdf = np.ma.filled(cdf_m3,0).astype('uint8')
    G = cdf[g]
    output=cv2.merge((B,G,R))                                         # Merging all the single channeled array to multichanneled array i.e the colured image
    return True, output
      
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.png"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
	
   # Write low pass filter here   
   
   img_out =img_in                                                    #Take the image as input
   img_out=cv2.cvtColor(img_out,cv2.COLOR_BGR2GRAY)		              #Convert the color image to Gray image
   dft = cv2.dft(np.float32(img_out),flags = cv2.DFT_COMPLEX_OUTPUT)  #Converting the image in floating point array and get a complex output array i.e Dft of image
   dft_shift = np.fft.fftshift(dft)                                   #Shift the zero component to center of spectrum and applies shift to all axes
   rows, cols = img_out.shape[0],img_out.shape[1]                     #Get the width and height of the image
   hrow,hcol = rows/2 , cols/2                                        
   mask = np.zeros((rows,cols,2),np.uint8)                            #Creating a Zero array of size rows*cols
   mask[hrow-10:hrow+10, hcol-10:hcol+10] = 1                         #Setting The center bits of mask to 1
   fshift = dft_shift*mask                                            #applying mask to the Dft of image
   f_ishift = np.fft.ifftshift(fshift)                                #Shifting the Zero component to its original position and applies shift to all axes
   img_back = cv2.idft(f_ishift,flags=cv2.DFT_SCALE)                  #Inverse the Dft and get a complex array and scale back to original size
   img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])          #Calculate the magnitude of each array element

   return True, img_back

def high_pass_filter(img_in):

   # Write high pass filter here 
   img_out =img_in						                              #Take the image as input
   img_out=cv2.cvtColor(img_out,cv2.COLOR_BGR2GRAY)		              #Convert the color image to Gray image
   f = np.fft.fft2(img_out)					                          #Get a frequency transform of image and get complex array
   fshift = np.fft.fftshift(f)                                        #Shift the zero component to center of spectrum and applies shift to all axes
   rows, cols = img_out.shape[0],img_out.shape[1]		      		  #Get the width and height of the image
   hrow,hcol = rows/2 , cols/2                                        #Get the width and height of the image and divide by 2
   fshift[hrow-10:hrow+10, hcol-10:hcol+10] = 0			              #Setting The center bits of mask to 0
   f_ishift = np.fft.ifftshift(fshift)  			      			  #Shifting the Zero component to its original position and applies shift to all axes
   img_back = np.fft.ifft2(f_ishift)                                  #Inverse the Fft and get a complex array 
   img_back = np.abs(img_back)                                        #Calculate the absolute value of each array element
   return True, img_back
   
def deconvolution(img_in):
   
   # Write deconvolution codes here
   img_out = img_in    						                      #Calculate the magnitude of array
   gk = cv2.getGaussianKernel(21,5)				  				  #Get the Gausiasian kernel
   gk = gk * gk.T                                                 #Get the transpose and muliplying with kernel.

   def ft(img_out, newsize=None):
       dft = np.fft.fft2(np.float32(img_out),newsize)             #Get a frequency transform of image and get complex array of new size
       return np.fft.fftshift(dft)                                #Shift the zero component to center of spectrum and applies shift to all axes


   def ift(shift):
       f_ishift = np.fft.ifftshift(shift)                         #Shifting the Zero component to its original position and applies shift to all axes
       img_back = np.fft.ifft2(f_ishift)                          #Inverse the Fft and get a complex array
       return np.abs(img_back)

   imf = ft(img_out, (img_out.shape[0],img_out.shape[1]))         #Getting the frequency transform of image and getting transform array of size of image
   gkf = ft(gk, (img_out.shape[0],img_out.shape[1]))              #Getting the frequency transform of gausian kernel and getting transform array of size of image
   imdeconvf = imf / gkf                                          #Calculating the denconvolution array
  
   deconvimg= ift(imdeconvf)                                      #Calculating the IFT from Deconvolution matrix of image
   deconvimg=deconvimg*255                                        #Scaling the Image back to original pixels

   return True, deconvimg

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)   
   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2LPF.png"
   output_name2 = sys.argv[4] + "2HPF.png"
   output_name3 = sys.argv[4] + "2deconv.png"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

    A1=img_in1                                          #Take the image1 as input
    B1=img_in2											#Take the image2 as input

    A1= A1[:,:A1.shape[0]]                              #Make image rectangular
    B1 = B1[:A1.shape[0],:A1.shape[0]]                  #Make images of same shape
   
    

    GA = A1.copy()                                      #Create a shallow copy of Image Matrix A1
    gpA = [GA]                                          #Assign value of GA to list gpA                                      
    for i in xrange(6):
        GA = cv2.pyrDown(GA)                            #Blurring an image and downsampling it level by level.
        gpA.append(GA)                                  
    
    GB = B1.copy()										#Create a shallow copy of Image Matrix Bi
    gpB = [GB]											#Assign value of GB to list gpB
    for i in xrange(6):  
        GB = cv2.pyrDown(GB)                            #Blurring an image and downsampling it level by level.
        gpB.append(GB)

    lpA = [gpA[5]]                                      #Assign the value of last level of Gaussian pyramid to lpA
    for i in xrange(5,0,-1):
        GEA = cv2.pyrUp(gpA[i])                         #Upsampling an image and blurring it
        LA = cv2.subtract(gpA[i-1],GEA)                 #Subtracting the blurred gaussian pyramid level image and laplacian pyramid level of image
        lpA.append(LA)                                  #Appending subtract result to a list

    lpB = [gpB[5]]					                    #Assign the value of last level of Gaussian pyramid to lpB
    for i in xrange(5,0,-1):
        GEB = cv2.pyrUp(gpB[i])                         #Upsampling an image and blurring it 
        LB = cv2.subtract(gpB[i-1],GEB)			        #Subtracting the blurred gaussian pyramid level image and laplacian pyramid level of image
        lpB.append(LB)					                #Appending subtract result to a list

    LS = []                                             
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape                        #Getting Image Size 
    	ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:])) #Stacking the both image arrays horizontally side by side columwise
        LS.append(ls)                                   #Appending the merged array to list


    ls_ = LS[0]                                         #Getting the first level of merging
    for i in xrange(1,6):
        ls_ = cv2.pyrUp(ls_)                            #Upsampling an image at level and blurring it
        ls_ = cv2.add(ls_, LS[i])                       #Adding the upsampled array leevl by level to merged array to create an image

    return True,ls_
def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "3.png"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
