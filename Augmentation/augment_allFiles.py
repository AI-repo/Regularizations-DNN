#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.pylab as pylab
import matplotlib.image as matimage
import cv2

import numpy as np
#matplotlib inline
import matplotlib.image as mpimg
import scipy.misc
import os, sys, shutil
import random
from skimage import io
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

NUMBER_AUGMENTED_SAMPLES = 9 #100

def elastic_transform(image, alpha, sigma, spline_order, mode, random_state=np.random):
     
    assert image.shape[2] == 3
    shape = image.shape[:2]
    if random_state is None:
        random_state = np.random.RandomState(None)

   
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
 
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))]
    
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:,:,i] = map_coordinates(image[:,:,i], indices, order=spline_order, mode=mode).reshape(shape)
    
    return result

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
    
def transform_image(img,ang_range,shear_range,trans_range,brightness=1):
    
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    shape_len = len(img.shape)
    if shape_len == 3:
        rows,cols,ch = img.shape  
    else:
        rows,cols = img.shape  

    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    img = elastic_transform(img , 120,10 , 2 ,'nearest', np.random)
    
    # Brightness
    if brightness == 1 & shape_len == 3:
      img = augment_brightness_camera_images(img)

    return img
    
#def augment_image_with_plot(img_name):        
#    #image = mpimg.imread('stopsign.jpg')
#    image = mpimg.imread(img_name)
#    plt.imshow(image);
#    plt.axis('off');
#    
#    gs1 = gridspec.GridSpec(10, 10)
#    gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
#    plt.figure(figsize=(12,12))
#    for i in range(NUMBER_AUGMENTED_SAMPLES):
#        ax1 = plt.subplot(gs1[i])
#        ax1.set_xticklabels([])
#        ax1.set_yticklabels([])
#        ax1.set_aspect('equal')
#        img = transform_image(image,20,10,5,brightness=1)
#        img_split = img_name.split('.')
#        print(img_split)
#        #scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save(img_split[0]+str(i)+'.'+img_split[1])
#        img_save_path = subdir_path2+'train/'+img_split[0]+'_'+str(i)+'.'+img_split[1]
#                        
#        print(img_save_path)
#        matimage.imsave(img_save_path, img)
#
#        plt.subplot(10,10,i+1)
#        plt.imshow(img)
#        plt.axis('off')
#    
#    plt.show()
        
def augment_image(img_name):  
    image = mpimg.imread(img_name)
    #image = io.imread(img_name, as_grey=True)
    
    for i in range(NUMBER_AUGMENTED_SAMPLES):
        img = transform_image(image,10,4,3,brightness=1)
        
        #Save path train
        img_split = img_name.split('/')
        print(img_split)
        #img_save_path = subdir_path2+'train/'+img_split[0]+'_'+str(i)+'.'+img_split[1]
        img_save_path = ''
        for j in range(0, len(img_split)):
            if (j != len(img_split)-1):
                img_save_path = img_save_path+img_split[j]+'/'
            else:
                img_name_split = img_split[j].split('.')
#                img_save_path = img_save_path+'train1/'+img_name_split[0]+'_'+str(i)+'.'+img_name_split[1]
                img_save_path = img_save_path+ img_name_split[0]+'_'+str(i)+'.'+img_name_split[1]
                
        print(img_save_path)
        matimage.imsave(img_save_path, img)

def augment_images(files, subdir_path2): 
    print(files)
    for k in range(0, len(files)): 
        if (".jpg" in files[k]):
            files_path = subdir_path2+files[k]
            print(files_path)
            augment_image(files_path)

def separate_images(subdir_path1):    
    train_path = subdir_path1+'train/'
    print(train_path)
    files = os.listdir(train_path)
    print(os.path.exists(train_path))
            
    train_number = int(round(len(files)*0.75))
    test_number = len(files) - train_number        
    print('Total: '+str(len(files))+', Train: '+str(train_number)+', Test: '+str(test_number))
    
    test_path = subdir_path1+'test1/'
    print(test_path)
    print(os.path.exists(test_path))
    group_of_items = files # a sequence or set will work here.
    num_to_select = test_number # set the number to select here.
    list_of_random_items = random.sample(group_of_items, num_to_select)
    print(list_of_random_items)    
    
    #Move 25% of items from train folder to test folder
    for f in list_of_random_items:
        src = train_path+f
        dst = test_path+f
        shutil.move(src,dst)

def separate_images2(subdir_path2):    
    files = os.listdir(subdir_path2)
    files2 = []
    for f in files:
        if ('.jpg' in f):
            files2 = files2 + [f]
    print(files2)
    train_number = int(round(len(files2)*0.75))
    test_number = len(files2) - train_number        
    
    train_path = subdir_path2
    test_path = subdir_path2+'test/'
    group_of_items = files2 # a sequence or set will work here.
    num_to_select = test_number # set the number to select here.
    list_of_random_items = random.sample(group_of_items, num_to_select)
    print(list_of_random_items)
    
    #Move 25% of items from train folder to test folder
    #ATTENTION: moving files from root folder to test
    for f in list_of_random_items:
        src = train_path+f
        dst = test_path+f
        shutil.move(src,dst)    
    
if __name__ == "__main__":
    
    main_dir = 'data/'
    #Namelist of the files in the directory 
    main_dir_namelist = os.listdir(main_dir)
    #In Mac, the first directory is .DS_Store, so we start from index 1    
    print(main_dir_namelist)
    for i in range(0, len(main_dir_namelist)):
        subdir_path1 = main_dir+main_dir_namelist[i]+'/'        
#        subdir_path1_namelist= os.listdir(subdir_path1)
#        print(subdir_path1_namelist)

#         for j in range(0, len(subdir_path1_namelist)):
#             subdir_path2 = main_dir+main_dir_namelist[i]+'/'+subdir_path1_namelist[j]+'/'

        subdir_path2 = main_dir+main_dir_namelist[i]+'/'+'train/'
#        train_dir = subdir_path1+'train1/'
#
#        if not os.path.exists(train_dir):
#            os.makedirs(train_dir)
#        else:
#            shutil.rmtree(train_dir) #removes all the subdirectories!
#            os.makedirs(train_dir)
   
        test_dir = subdir_path1+'test1/'
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        else:
            shutil.rmtree(test_dir) #removes all the subdirectories!
            os.makedirs(test_dir)
              
            
        #Run through all images
        #files = os.listdir(subdir_path2)
        files = os.listdir(subdir_path2)            
        jpg_numbers = 0
        for k in range(0, len(files)):
            if (".jpg" in files[k]):
                jpg_numbers = jpg_numbers+1
        
        #TO-DO: separate into 25% test and 75% train
        if (jpg_numbers < 8): #augment -> separate
            #print('Images < 5')    
            augment_images(files, subdir_path2)
            separate_images(subdir_path1)
        #else: #separate -> augment
            #print('Images >= 5')    
            #separate_images2(subdir_path2)
            #files = os.listdir(subdir_path2)
            #augment_images(files, subdir_path2)
            
        #Move rest of images from root to train folder
#==============================================================================
    src_path = subdir_path2
    #dst_path = subdir_path2+'train/'
    dst_path = train_dir    
    files = os.listdir(subdir_path2)
    for f in files:
        if ('.jpg' in f):
            src = src_path+f
            dst = dst_path+f
            shutil.move(src,dst)    
#==============================================================================
    
    