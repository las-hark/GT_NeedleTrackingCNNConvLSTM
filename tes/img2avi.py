import cv2  # 3.4.2
import os

fps = 30  # 视频帧率
# fourcc = cv2.CV_FOURCC('M', 'J', 'P', 'G')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter('demo_256.avi', fourcc, fps,(384,384))  
#!!!!!!!!改名字！！！！！！！！！！
img_names = os.listdir(r"E:\US\USPP")
img_names.sort()
for i in range(0, len(img_names)):
    img_path = os.path.join(r"E:\US\USPP", img_names[i])
    img = cv2.imread(img_path)
    cv2.imshow('img', img)
    # cv2.waitKey(1000/int(fps))
    videoWriter.write(img)
# for i in range(1245, len(img_names)):
#     img_path = os.path.join(r"E:\US\USPP", img_names[1799-(i-1245)-1])
#     img = cv2.imread(img_path)
#     # cv2.imshow('img', img12)
#     # cv2.waitKey(1000/int(fps))
#     videoWriter.write(img)
videoWriter.release()
print("finish\n")

