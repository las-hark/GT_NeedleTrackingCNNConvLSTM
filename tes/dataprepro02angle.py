import os,cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters,io,color
from skimage.morphology import disk
# Define the paths
dataset_path0 = r"E:\US\dataset01"
annotations_path0 = r"E:\US\annotation01"
startflags0 = [10001,11245,20001,21281,30001,31245,40001,41245,50126,51387]
endflags0   = [10380,11589,20489,21675,30612,31793,40612,41793,50612,51793]
# Define the paths
dataset_path1 = r"E:\US\dataset02"
annotations_path1 = r"E:\US\annotation02"
startflags1 = [10001,11245,20071,21338,30001,31245,40001,41245,50135,51387]
endflags1   = [10354,11587,20442,21638,30593,31763,40593,41763,50593,51763]
# Define the paths
dataset_path2 = r"E:\US\dataset03"
annotations_path2 = r"E:\US\annotation03"
startflags2 = [10001,11245,20127,21378,30001,31245,40001,41245,50135,51387]
endflags2   = [10338,11570,20412,21628,30576,31730,40580,41730,50595,51760]
# Define the paths
dataset_path3 = r"E:\US\dataset04"
annotations_path3 = r"E:\US\annotation04"
startflags3 = [10001,11245,20130,21381,30001,31245,40001,41245,50135,51387]
endflags3   = [10328,11565,20393,21610,30552,31708,40556,41712,50590,51752]
# Define the paths
dataset_path4 = r"E:\US\dataset05"
annotations_path4 = r"E:\US\annotation05"
startflags4 = [10001,11245,20125,21378,30001,31276,40001,41265,50142,51430]
endflags4   = [10328,11570,20378,21589,30530,31691,40552,41706,50592,51764]

nump = 0
for i in range(0,10):
    nump += endflags0[i] - startflags0[i] + 1
    nump += endflags1[i] - startflags1[i] + 1
    nump += endflags3[i] - startflags3[i] + 1
    nump += endflags4[i] - startflags4[i] + 1
    nump += endflags2[i] - startflags2[i] + 1
# Load the dataset images and annotations
num_frames = 5
frame_shape = (256,256)
images_arr = np.zeros((5000,num_frames,*frame_shape), dtype=np.float32)
annotations_arr = np.zeros((5000,2), dtype=np.float32)

def loaddata(dataset_path,annotations_path,startflag,endflag):
    # Load the dataset images and annotations
    images = []
    annotations = []
    end = endflag + 1
    for filename in os.listdir(dataset_path):
        No_file  = int(filename[:-4])
        # if filename in useless_data:
        #     continue
        if No_file == end:
            break
        if No_file >= startflag:
            if filename.endswith('.jpg'):
                # Load the image
                image_path = os.path.join(dataset_path, filename)
                image = io.imread(image_path)
                image = color.rgb2gray(image)
                images.append(image)

                # Load the annotation
                annotation_path = os.path.join(annotations_path, filename[:-4] + '.txt')
                with open(annotation_path, 'r') as f:
                    annotation = f.readline().strip().split()
                    annotation = [float(coord) for coord in annotation]
                    annotations.append(annotation)

    # Convert the images and annotations to numpy arrays
    images = np.array(images)
    annotations = np.array(annotations)
    imagess = np.zeros((len(images)-1,256,256))
    for imgdic in range(0,len(images)-1):
        picform = images[imgdic]
        picused = images[imgdic+1]
        picenha = picused-picform
        picenhb = filters.median(picenha,disk(4))
        picres  = picused + picenhb
        imagess[imgdic] = picres
    annotationss = annotations[1:,:]
    flagt = len(annotationss) - 4
    return (imagess,annotationss,flagt)  

ind_start = 0
ind_end   = 0
for i in range(0,len(startflags0)):
    (images_temp,annotations_temp,step) = loaddata(dataset_path0,annotations_path0,startflags0[i],endflags0[i])
    lenstep = len(images_temp[4:-1:5])
    ind_end   = ind_start + lenstep
    for j in range(0,5):
        No_img = 5-j
        images_arr[ind_start:ind_end,j] = images_temp[j:-No_img:5]
        annotations_arr[ind_start:ind_end] = annotations_temp[j:-No_img:5]
    ind_start = ind_start + lenstep +1

for i in range(0,len(startflags1)):
    (images_temp,annotations_temp,step) = loaddata(dataset_path1,annotations_path1,startflags1[i],endflags1[i])
    lenstep = len(images_temp[4:-1:5])
    ind_end   = ind_start + lenstep
    for j in range(0,5):
        No_img = 5-j
        images_arr[ind_start:ind_end,j] = images_temp[j:-No_img:5]
        annotations_arr[ind_start:ind_end] = annotations_temp[j:-No_img:5]
    ind_start = ind_start + lenstep +1

for i in range(0,len(startflags2)):
    (images_temp,annotations_temp,step) = loaddata(dataset_path2,annotations_path2,startflags2[i],endflags2[i])
    lenstep = len(images_temp[4:-1:5])
    ind_end   = ind_start + lenstep
    for j in range(0,5):
        No_img = 5-j
        images_arr[ind_start:ind_end,j] = images_temp[j:-No_img:5]
        annotations_arr[ind_start:ind_end] = annotations_temp[j:-No_img:5]
    ind_start = ind_start + lenstep +1

for i in range(0,len(startflags3)):
    (images_temp,annotations_temp,step) = loaddata(dataset_path3,annotations_path3,startflags3[i],endflags3[i])
    lenstep = len(images_temp[4:-1:5])
    ind_end   = ind_start + lenstep
    for j in range(0,5):
        No_img = 5-j
        images_arr[ind_start:ind_end,j] = images_temp[j:-No_img:5]
        annotations_arr[ind_start:ind_end] = annotations_temp[j:-No_img:5]
    ind_start = ind_start + lenstep +1

for i in range(0,len(startflags4)):
    (images_temp,annotations_temp,step) = loaddata(dataset_path4,annotations_path4,startflags4[i],endflags4[i])
    lenstep = len(images_temp[4:-1:5])
    ind_end   = ind_start + lenstep
    for j in range(0,5):
        No_img = 5-j
        images_arr[ind_start:ind_end,j] = images_temp[j:-No_img:5]
        annotations_arr[ind_start:ind_end] = annotations_temp[j:-No_img:5]
    ind_start = ind_start + lenstep +1            

images_arr0 = images_arr[0:ind_end]
annotations_arr0 = annotations_arr[0:ind_end]
# Split the dataset into training,validation and test sets
num_samples = len(images_arr0)
indices = np.arange(num_samples)
np.random.shuffle(indices)

split_idx = int(num_samples * 0.85)
train_tv_indices = indices[:split_idx]
test_indices = indices[split_idx:]

split_tv_idx = int(len(train_tv_indices)*0.8)
train_indices = train_tv_indices[:split_tv_idx]
val_indices = train_tv_indices[split_tv_idx:]

x_train = images_arr0[train_indices]
y_train = annotations_arr0[train_indices]
x_val   = images_arr0[val_indices]
y_val   = annotations_arr0[val_indices]
x_test  = images_arr0[test_indices]
y_test  = annotations_arr0[test_indices]

#save the dataset
np.save(r'E:\US\x_trains.npy',x_train)
np.save(r'E:\US\y_trains.npy',y_train)
np.save(r'E:\US\x_valids.npy',x_val)
np.save(r'E:\US\y_valids.npy',y_val)
np.save(r'E:\US\x_testds.npy',x_test)
np.save(r'E:\US\y_testds.npy',y_test)
a = 0