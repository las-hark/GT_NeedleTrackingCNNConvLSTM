import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import (denoise_wavelet)
from skimage import io,color,filters,morphology,measure,transform
from skimage import img_as_ubyte
from skimage.morphology import disk
import cv2

img1 = io.imread(r"E:\US\USPP\01456.jpg")
img1 = np.clip(img1,1,255)
img1 = color.rgb2gray(img1)
#------------------------------------------------------------------------------------------------
#wavelet denoise#
im = np.log(img1) #取对数
im_bayes = denoise_wavelet(im,wavelet="bior6.8",wavelet_levels=3,
                           method='BayesShrink', mode='soft')
img11 = np.exp(im_bayes) #还原

img2 = io.imread(r"E:\US\USPP\01455.jpg")
img2 = np.clip(img2,1,255)
img2 = color.rgb2gray(img2)
#------------------------------------------------------------------------------------------------
#wavelet denoise#
im2 = np.log(img2) #取对数
im_bayes2 = denoise_wavelet(im2,wavelet="bior6.8",wavelet_levels=3,
                           method='BayesShrink', mode='soft')
img21 = np.exp(im_bayes2) #还原
imgres = img2-img1
imgres2 = filters.median(imgres,disk(4))
imgeha = imgres+img2
# plt.imshow(imgres*8)
# plt.show()
# fig, axes = plt.subplots(1, 3, figsize=(8, 8))
# ax = axes.ravel()
# ax[0].imshow(imgres)
# ax[0].set_title("(a)")
# ax[0].axis('off')
# ax[1].imshow(imgres2)
# ax[1].set_title("(b)")
# ax[1].axis('off')
# ax[2].imshow(imgeha)
# ax[2].set_title("(c)")
# ax[2].axis('off')
# plt.show()
####debug下一行还没改此处都是归一化后的矩阵应该是需要重新映射到0~255的不然做个球诶哟卧槽泥马杀人了####
# imgres = np.uint8(np.floor(255*img11)) & np.uint8(np.floor(255*img21))

# ####debug：opencv.ver问题出在小波没效果他妈的杀人了
# img1 = cv2.imread(r"E:\US\USPP\01500.jpg",0)
# img2 = cv2.imread(r"E:\US\USPP\01501.jpg",0)
# ##如果不去噪会出现散斑噪声如imge
# #imge =  img2 & img1
# #------------------------------------------------------------------------------------------------
# #wavelet denoise#
# im1 = np.log(img1) #取对数
# im_bayes1 = denoise_wavelet(im1,wavelet="bior6.8",wavelet_levels=3,
#                            method='BayesShrink', mode='soft')
# img_a1 = np.round(np.exp(im_bayes1)) #还原
# img_a1 = img_a1.astype(np.uint8)

# im2 = np.log(img2) #取对数
# im_bayes2 = denoise_wavelet(im2,wavelet="bior6.8",wavelet_levels=3,
#                            method='BayesShrink', mode='soft')
# img_a2 = np.exp(im_bayes2) #还原
# img_a2 = img_a2.astype(np.uint8)
# imgres= cv2.bitwise_and(img_a1,img_a2)
testpara = 1