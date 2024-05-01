import cv2 
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import (denoise_wavelet)
from skimage import io,color,filters,morphology
from skimage import img_as_ubyte
from skimage.morphology import disk

img_names = os.listdir(r"E:\US\USPP")
img_names.sort()

for i in range(0, len(img_names)):
    img_path = os.path.join(r"E:\US\USPP", img_names[i])
    img_savepath = os.path.join(r"E:\US\USPPro", img_names[i])
    img1 = io.imread(img_path)
    img1 = np.clip(img1,1,255)
    img1 = color.rgb2gray(img1)
    #------------------------------------------------------------------------------------------------
    #wavelet denoise#
    im = np.log(img1) #取对数
    im_bayes = denoise_wavelet(im,wavelet="bior6.8",wavelet_levels=3,
                               method='BayesShrink', mode='soft')
    img = np.exp(im_bayes) #还原
    #------------------------------------------------------------------------------------------------
    #binary#
    med = filters.median(img,disk(5)) #中值滤波 3改成了5
    thresh = filters.threshold_triangle(med) #生成阈值
    binary = med > thresh #生成布尔数组
    # binary = img_as_ubyte(binary)
    # imge = morphology.erosion(binary,disk(4))
    # imgr = morphology.dilation(imge,disk(3))
    # # dst = morphology.opening(binary,disk(3)) 
    # # imgrr = img*imgr
    # skeleton =morphology.skeletonize(imgr)
    # skeleton = morphology.dilation(skeleton,disk(2))
    imgres = morphology.remove_small_objects(binary,min_size=400)
    imgres= img_as_ubyte(imgres)
    io.imsave(img_savepath,imgres) 
    # img1 = []
    # im = []
    # im_bayes = []
    # img = []
    # med = []
    # thresh = 0
    # binary = []
    # dst = []

print("finish\n")