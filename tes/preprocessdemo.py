import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import (denoise_wavelet)
from skimage import io,color,filters,morphology,measure,transform
from skimage import img_as_ubyte
from skimage.morphology import disk
import time

start = time.time()
img1 = io.imread(r"M:\00234.jpg")
img1 = color.rgb2gray(img1)
#------------------------------------------------------------------------------------------------
#wavelet denoise#
im = np.log(img1) #取对数
im_bayes = denoise_wavelet(im,wavelet="bior6.8",wavelet_levels=3,
                           method='BayesShrink', mode='soft')
img = np.exp(im_bayes) #还原
# #输出去除的噪声
# tmp=img1-img
# plt.imshow(tmp,cmap=plt.cm.gray)
# plt.axis('off')
# plt.show()
# #输出前后图像
# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# ax = axes.ravel()
# ax[0].imshow(img1,cmap=plt.cm.gray)
# ax[0].set_title("before")
# ax[0].axis('off')
# ax[1].imshow(img,cmap=plt.cm.gray)
# ax[1].set_title("after")
# ax[1].axis('off')
# ax[2].imshow(im,cmap=plt.cm.gray)
# ax[2].set_title("logbefore")
# ax[2].axis('off')
# ax[3].imshow(im_bayes,cmap=plt.cm.gray)
# ax[3].set_title("logafter")
# ax[3].axis('off')
# fig.tight_layout()
# plt.show()
#------------------------------------------------------------------------------------------------
#binary#
dst = filters.median(img,disk(5))
# #morphology.opening(med,disk(1))
# fig, axes = plt.subplots(1, 2, figsize=(8, 8))
# ax = axes.ravel()
# ax[0].imshow(img,cmap=plt.cm.gray)
# ax[0].axis('off')
# ax[0].set_title("before")
# ax[1].imshow(dst,cmap=plt.cm.gray)
# ax[1].set_title("after")
# plt.axis('off')
# fig.tight_layout()

# plt.show()

# plt.hist(dst.ravel(), 256)
# plt.show()

# # Otsu
# thresh_otsu = filters.threshold_otsu(dst)
# binary_otsu = dst > thresh_otsu
# binary_otsu = img_as_ubyte(binary_otsu)
# # Niblack
# thresh_niblack = filters.threshold_niblack(dst, window_size=25, k=0.8)
# binary_niblack = dst > thresh_niblack
# binary_niblack = img_as_ubyte(binary_niblack)
# # Sauvola
# thresh_sauvola = filters.threshold_sauvola(dst, window_size=25)
# binary_sauvola = dst > thresh_sauvola
# binary_sauvola = img_as_ubyte(binary_sauvola)
# # triangle
# thresh_triangle = filters.threshold_triangle(dst)
# binary_triangle = dst > thresh_triangle
# binary_triangle = img_as_ubyte(binary_triangle)
# fig, axes = plt.subplots(2, 2)
# ax = axes.ravel()
# ax[0].imshow(binary_otsu,cmap=plt.cm.gray)
# ax[0].set_title("otsu")
# ax[0].axis('off')
# ax[1].imshow(binary_niblack,cmap=plt.cm.gray)
# ax[1].set_title("niblack") 
# ax[1].axis('off')
# ax[2].imshow(binary_sauvola,cmap=plt.cm.gray)
# ax[2].set_title("sauvole")
# ax[2].axis('off')
# ax[3].imshow(binary_triangle,cmap=plt.cm.gray)
# ax[3].set_title("triangle") 
# ax[3].axis('off')

# plt.show()

thresh = filters.threshold_triangle(dst) #生成阈值
binary = (dst > thresh)/255 #生成布尔数组
# binary = img_as_ubyte(binary)
imgr0 = morphology.erosion(binary,disk(4))
imgres = morphology.dilation(imgr0,disk(3)) #二值化后开运算
skeleton =morphology.skeletonize(imgres)
skeleton = morphology.dilation(skeleton,disk(2))
# fig, axes = plt.subplots(1, 2)
# ax = axes.ravel()
# ax[0].imshow(img1,cmap=plt.cm.gray)
# ax[0].set_title("before")
# ax[0].axis('off')
# ax[1].imshow(skeleton,cmap=plt.cm.gray)
# ax[1].set_title("open")
# ax[1].axis('off')
# ax[2].imshow(skeleton,cmap=plt.cm.gray)
# ax[2].set_title("skeleton")
# ax[2].axis('off')
# plt.show()
imglabel = measure.label(imgres)
imgregion = measure.regionprops(imglabel)
#连通域矩形边长，连通域离离心率
#需要综合考虑几个因素，权值如何确定
#hough变换
region_area = [] #面积
region_rect = [] #长方形
region_minr = [] #最小矩形
fig = plt.figure()
for i in range(np.max(imglabel)):
    region_area.append(imgregion[i].area)
    region_rect.append(imgregion[i].bbox)
end =time.time()

a = 1
# imgrr = img*imgr 
# fig, axes = plt.subplots(1, 4)
# ax = axes.ravel()
# ax[0].imshow(img,cmap=plt.cm.gray)
# ax[0].set_title("ori")
# ax[1].imshow(dst,cmap=plt.cm.gray)
# ax[1].set_title("median") #中值滤波
# ax[2].imshow(binary,cmap=plt.cm.gray)
# ax[2].set_title("binary") #二值
# ax[3].imshow(imgrr,cmap=plt.cm.gray)
# ax[3].set_title("mask") #开运算后掩膜
# fig.tight_layout()
# 
plt.imshow(imglabel*20,camp=plt.cm.gray)
plt.show()



