import os,cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters,io,color
from skimage.morphology import disk
# Define the paths
dataset_path = r"E:\US\dataset01"
annotations_path = r"E:\US\annotation01"

# # Define a list of filenames to skip
# arr = list(str(i0) for i0 in range(10381,11454))
# arr = arr+list(str(i) for i in range(20490,21368))
# arr = arr+list(str(i) for i in range(21764,21799))
# arr = arr+list(str(i) for i in range(30613,31244))
# arr = arr+list(str(i) for i in range(40613,41244))
# arr = arr+list(str(i) for i in range(50001,50125))
# arr = arr+list(str(i) for i in range(50613,51244))
# arr = arr+list(str(i) for i in range(51658,51799))
# useless_data = []
# for k in range(0,len(arr)):
#     fullname = str(arr[k]) + '.jpg'
#     useless_data.append(fullname) 
#     fullname = []

# splitflags = [11455,11799,21369,21763,31245,31799,41245,41799,51245,51657]
startflags = [10001,11245,20001,21281,30001,31245,40001,41245,50126,51245]
endflags   = [10380,11589,20489,21675,30612,31793,40612,41793,50612,51793]
# for i in range(0,len(endflags)):
#     endflags[i] = endflags[i] - 4
nump = 0
for i in range(0,10):
    nump += endflags[i] - startflags[i] + 1
# Load the dataset images and annotations
num_frames = 5
frame_shape = (256,256)
images_arr = np.zeros((4916,num_frames,*frame_shape), dtype=np.float32)
annotations_arr = np.zeros((4916,2), dtype=np.float32)

def loaddata(startflag,endflag):
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

images_fin = []
annotations_fin = []
images_fin = np.array(images_fin)
annotations_fin = np.array(annotations_fin)
ind_start = 0
ind_end   = 0
for i in range(0,len(startflags)):
    (images_temp,annotations_temp,step) = loaddata(startflag=startflags[i],endflag=endflags[i])
    ind_end   = ind_start + step -1
    for j in range(0,5):
        No_img = 5-j
        images_arr[ind_start:ind_end,j] = images_temp[j:-No_img]
        annotations_arr[ind_start:ind_end] = annotations_temp[j:-No_img]
    ind_start = ind_start + step

# Split the dataset into training,validation and test sets
num_samples = len(images_arr)
indices = np.arange(num_samples)
np.random.shuffle(indices)

split_idx = int(num_samples * 0.85)
train_tv_indices = indices[:split_idx]
test_indices = indices[split_idx:]

split_tv_idx = int(len(train_tv_indices)*0.8)
train_indices = train_tv_indices[:split_tv_idx]
val_indices = train_tv_indices[split_tv_idx:]

x_train = images_arr[train_indices]
y_train = annotations_arr[train_indices]
x_val   = images_arr[val_indices]
y_val   = annotations_arr[val_indices]
x_test  = images_arr[test_indices]
y_test  = annotations_arr[test_indices]

#save the dataset
np.save(r'E:\US\x_train.npy',x_train)
np.save(r'E:\US\y_train.npy',y_train)
np.save(r'E:\US\x_valid.npy',x_val)
np.save(r'E:\US\y_valid.npy',y_val)
np.save(r'E:\US\x_testd.npy',x_test)
np.save(r'E:\US\y_testd.npy',y_test)
a = 0