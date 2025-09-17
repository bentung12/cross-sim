import os
from PIL import Image
import numpy as np 
import pandas as pd
import pickle
import array 
import sys

path = "../../../data/datasets/tiny-imagenet-200/train/"
files = folders = 0
folder_names = []
dirlist = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path,item))]

GreyScaleImgs = []
endpathconstant = '/images/'
endpathconstant_val = '/images/'
num_classes = 200
N_train_total = 100000
N_val = 10000
train_dir = "../../../data/datasets/tiny-imagenet-200/train/"
val_dir = "../../../data/datasets/tiny-imagenet-200/val/"
test_dir = "../../../dnn/data/datasets/tiny-imagenet-200/test/"
train_dir_orig = "../../../data/datasets/tiny-imagenet-200/train/"



####PUTTING A KEY FOR THE VALIDATION THIS MOST LIKELY WORKS#####

df_val =pd.read_table(val_dir + 'val_annotations.txt',header=None)
df_val.columns = ["Image","Label","dunno0","dunno1","dunno2","dunno3"]
y_val = {}
y_val_int = {}
for dirname in range(len(dirlist)):
    #ylabel_list = []
    #labelname = dirlist[dirname]
    #assigns integer value as key 
    #y_val[dirlist[dirname]] = dirlist[dirname]
    #assigns name as key
    y_val[dirlist[dirname]] = dirlist[dirname]
#print(y_val)
    list_of_img = []
    for ind in df_val.index:
        #y_val[labelname]
        #list_of_img = []
        if(df_val['Label'][ind] == y_val[dirlist[dirname]]):

            list_of_img.append(df_val['Image'][ind]) 
        #if df_val['Label'][ind] in y_val:
         #   list_of_img.append(df_val['Image'][ind])
    #assigns integer value to label in dictionary     
    y_val_int[dirname] = list_of_img
    #assings string value of label in dictrionary 
    y_val[dirlist[dirname]] = list_of_img                        

print(dirlist)
# sd

###KEY FOR TEST IMAGE SET###

df_test =pd.read_table(val_test + 'val_annotations.txt',header=None)
df_val.columns = ["Image","Label","dunno0","dunno1","dunno2","dunno3"]
y_val = {}
y_val_int = {}
for dirname in range(len(dirlist)):
    #ylabel_list = []
    #labelname = dirlist[dirname]
    #assigns integer value as key 
    #y_val[dirlist[dirname]] = dirlist[dirname]
    #assigns name as key
    y_val[dirlist[dirname]] = dirlist[dirname]
#print(y_val)
    list_of_img = []
    for ind in df_val.index:
        #y_val[labelname]
        #list_of_img = []
        if(df_val['Label'][ind] == y_val[dirlist[dirname]]):

            list_of_img.append(df_val['Image'][ind]) 
        #if df_val['Label'][ind] in y_val:
         #   list_of_img.append(df_val['Image'][ind])
    #assigns integer value to label in dictionary     
    y_val_int[dirname] = list_of_img
    #assings string value of label in dictrionary 
    y_val[dirlist[dirname]] = list_of_img                        

print(dirlist)

###########YTEST##################
###Organizing files into numeric order###
#putting all validation images into a list
val_image_list = [f for f in os.listdir(val_dir + 'images/') if os.path.isfile(os.path.join((val_dir + 'images/'),f))]
#print('Val list', val_image_list)

#sorting the validation images list from 0 -> 9999
def get_number(image):
    return int(image.split("val_")[1].split('.JPEG')[0])

val_image_list_sorted = sorted(val_image_list, key=get_number)
#print('validation images sorted', val_image_list_sorted)    

#goes through sorted list and assigns label number based off of dictionary
val_image_list_sorted_INT = []
image_values_appended = []
for image in val_image_list_sorted:
    for key,value_list in y_val_int.items():
        if image in value_list:
            val_image_list_sorted_INT.append(key)
            image_values_appended.append(image)

#print(val_image_list_sorted_INT)
#print('length of y_val', len(val_image_list_sorted_INT))
y_test = val_image_list_sorted_INT



#####COLLECT TRAINING IMAGES PIXELS##### 
# get a list of subdirectories
subdirectories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d))]
#ytest 
#dont make a training for the two dataset just have trainig and test have the same keyset
x_train = np.zeros((N_train_total,64,64,3))
x_train_total = np.zeros((N_train_total,64,64,3))
#200 folders each holding 500 images associated with each label
#image format label_num.jpeg 

#EACH LABEL HAS ALL ASSOCIATED IMAGES IN THAT DIRECTORY 
#goes through each directory and extracts pixel data and assigns label data based off directory name 
image_count = 0
y_train  = []
for directory_name in subdirectories:
    train_dir_update = train_dir + directory_name + endpathconstant
    
    #assigns label integer based on folder descended 
    #THIS WILL MAKE 500 of same label integer in a row which is what you probably dont want 
    for j, (key,value_list) in enumerate(y_val.items()):
        if directory_name == key:
            index = j
            for _ in range(500):
                y_train.append(index)          
    #NOT SURE BOTTOM IS CORRECT ORIGINALY HAD IT GOING THROUGH 200 NEEDS TO GO THROUGH 500 
    for i in range(500): 
        
        x_i = np.asarray(Image.open(train_dir_update+directory_name+'_{:d}.JPEG'.format(i)))
        image_info = Image.open(train_dir_update+directory_name+'_{:d}.JPEG'.format(i))
        if image_info.mode != 'L':
            #x_train[(i + image_count),:,:,:] = x_i
            x_train[(i + image_count),:,:,:] = x_i
        else:
            #duplicates greyscale pixel value method
            #GreyScaleImgs.append(directory_name+'_{:d}.JPEG')
            #x_i_GreyToRGB = np.stack((x_i,)*3,axis=-1)
            #x_train[i,:,:,:] = x_i_GreyToRGB
            #changes picture from greyscale to RGB and then extracts
            img_greyscale = Image.open(train_dir_update+directory_name+'_{:d}.JPEG'.format(i))
            img_rgb = img_greyscale.convert("RGB")
            x_i_GreyToRGB =np.asarray(img_rgb)
            x_train[(i + image_count),:,:,:] = x_i_GreyToRGB
    
    
    image_count += 500
    train_dir_update = train_dir
    #train_dir = train_dir_orig
    #x_train_total = x_train_total + x_train
    #x_train_total.concatenate(x_train_total, x_train)
#print('Grey Scale Images in Dataset',GreyScaleImgs)
#print('total number of greyscale images', len(GreyScaleImgs))

zero_channel_mean = np.mean(x_train[:,:,:,0])
zero_channel_std = np.std(x_train[:,:,:,0])
one_channel_mean = np.mean(x_train[:,:,:,1])
one_channel_std = np.std(x_train[:,:,:,1])
two_channel_mean = np.mean(x_train[:,:,:,2])
two_channel_std = np.std(x_train[:,:,:,2])

x_train[:,:,:,0] = (x_train[:,:,:,0] - zero_channel_mean)/zero_channel_std
x_train[:,:,:,1] = (x_train[:,:,:,1] - one_channel_mean)/one_channel_std
x_train[:,:,:,2] = (x_train[:,:,:,2] - two_channel_mean)/two_channel_std


#####COLLECT VALIDATION IMAGES PIXELS#####
x_test = np.zeros((N_val,64,64,3))
#i=0
for img in range(len(val_image_list_sorted)):
        #y_i = np.asarray(Image.open(val_dir+endpathconstant_val+file_names[i]+'_{:d}.JPEG'.format(i)))
        y_i = np.asarray(Image.open(val_dir+endpathconstant_val+val_image_list_sorted[img].format(img)))
        image_info = Image.open(val_dir+endpathconstant_val+val_image_list_sorted[img].format(img))
        image_unpacked = val_image_list_sorted[img]
        if image_info.mode != 'L':
            x_test[img,:,:,:] = y_i
        else:
            GreyScaleImgs.append(val_dir+endpathconstant_val+val_image_list_sorted[img].format(img))
            #duplicates greyscale pixel value method
            #GreyScaleImgs.append(directory_name+'_{:d}.JPEG')
            #x_i_GreyToRGB = np.stack((x_i,)*3,axis=-1)
            #x_train[i,:,:,:] = x_i_GreyToRGB
            #changes picture from greyscale to RGB and then extracts
            img_greyscale = Image.open(val_dir+endpathconstant_val+val_image_list_sorted[img].format(img))
            img_rgb = img_greyscale.convert("RGB")
            y_img_GreyToRGB =np.asarray(img_rgb)
            x_test[img,:,:,:] = y_img_GreyToRGB
    #i+=1

###TASKS####
#mean of training set for that channel 
#use standard deviation of training set for that channel 
#use normalization factors for trainig and test set 
#compute and print and check against other tiny image net and make sure they match
#reducing normalization to reduce massive size



x_test[:,:,:,0] = (x_test[:,:,:,0] - zero_channel_mean)/zero_channel_std
x_test[:,:,:,1] = (x_test[:,:,:,1] - one_channel_mean)/one_channel_std
x_test[:,:,:,2] = (x_test[:,:,:,2] - two_channel_mean)/two_channel_std


#####ASSIGNING VALIDATION KEY TO IMAGE AND ASSOCIATING THAT FOR EACH IMAGE#####

#filenames = ["val_0.JPEG", "val_1.JPEG", ...., "val_9999.JPEG"]
#500 directories each has its own associated image value in it
#as one unpacks images in the 500 directories assigns integer label to each image unpacked based on name of folder




#####FINDING THE SHAPE OF EACH DATASET TYPE#######   
print("Xtrain shape", x_train.shape)
print("Xtest shape", x_test.shape)
print("y_train  LIST length", len(y_train ))
print("Y_test LIST length",len(y_test) )
#converting list to array
y_train  = np.array(y_train )
y_test = np.array(y_test)
print("Ytrain ARRAY shape", y_train.shape)
print("Ytest ARRAY shape", y_test.shape)

#####WRITING THE DATA SET TO FILES

# with open("x_train.txt","w") as data:
#     data.write(str(x_train)) 
# with open("y_train .txt","w") as data:
#     data.write(y_train ) 
# print("Ytrain shape", y_train.shape)
# with open("y_train.txt","w") as data:
#     data.write(y_train) 
# with open("y_test.txt","w") as data:
#     data.write(y_test) 
# y_train_8int = y_train.astype(np.uint8)
# x_train_8int = x_train.astype(np.uint8)
# y_test_8int = y_test.astype(np.uint8)
# x_test_8int = x_test.astype(np.uint8)

y_train_file_path = os.path.abspath("y_train.npy")
y_test_file_path = os.path.abspath("y_test.npy")
x_train_file_path = os.path.abspath("x_train.npy")
x_test_file_path = os.path.abspath("x_test.npy")


memsize_ytrain = sys.getsizeof(y_train)
print('Mem size of y_train 32bit', memsize_ytrain)
memsize_xtrain = sys.getsizeof(x_train)
print('Mem size of y_train 32bit', memsize_xtrain)
memsize_ytest = sys.getsizeof(y_test)
print('Mem size of y_train 32bit', memsize_ytrain)
memsize_xtest = sys.getsizeof(x_test)
print('Mem size of y_train 32bit', memsize_xtest)

# memsize_ytrain_8int = sys.getsizeof(y_train_8int)
# print('Mem size of y_train 8bit', memsize_ytrain_8int)
# memsize_xtrain_8int = sys.getsizeof(x_train_8int)
# print('Mem size of y_train 8bit', memsize_xtrain_8int)
# memsize_ytest_8int = sys.getsizeof(y_test_8int)
# print('Mem size of y_train 8bit', memsize_ytrain_8int)
# memsize_xtest_8int = sys.getsizeof(x_test_8int)
# print('Mem size of y_train 8bit', memsize_xtest_8int)


np.save(y_train_file_path,y_train)
np.save(y_test_file_path,y_test)
np.save(x_train_file_path,x_train)
np.save(x_test_file_path,x_test)

#np.load(file) 
#try rounding numbers in dataset to lower precision 

# with open('y_test.pkl', 'wb') as file:
#     pickle.dump(str(y_test), file)
# with open('y_train.pkl', 'wb') as file:
#     pickle.dump(str(y_train), file)
# with open('y_train.pkl', 'wb') as file:
#     pickle.dump(str(y_train ), file)
# with open('x_train.pkl', 'wb') as file:
#     pickle.dump(str(y_train), file)



