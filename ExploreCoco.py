from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator






import matplotlib
matplotlib.use('Agg')
from tkinter import *
import cv2
from PIL import ImageTk, Image
from  tkinter.filedialog import askopenfilename
from fastai.conv_learner import *
from fastai.dataset import *
from fastai.models.resnet import vgg_resnet50

import json





import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
dataDir='./COCOdataset2017'
dataType='train'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# Initialize the COCO api for instance annotations
coco=COCO(annFile)

# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats)


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

# Define the classes (out of the 81) which you want to see. Others will not be shown.

def GetPhotos () :
    # get all images containing given categories, select one at random
    filterClasses = ['person']
    catIds = coco.getCatIds(catNms=filterClasses);
    imgIds = coco.getImgIds(catIds=catIds);
    print("Number of images containing all required classes:", len(imgIds))
    # load and display image
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    I = io.imread('{}/images/{}/{}'.format(dataDir, dataType, img['file_name'])) / 255.0

    for q in range(len(imgIds)):
     img = coco.loadImgs(imgIds[q])[0]
     I = io.imread('{}/images/{}/{}'.format(dataDir, dataType, img['file_name'])) / 255.0
     img2 = cv2.convertScaleAbs(I, alpha=(255.0))
     x = 'E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/train/CreatedTrain/image' + str(q) + '.jpg'
     cv2.imwrite(filename=x, img=img2)
     annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
     anns = coco.loadAnns(annIds)
     filterClasses = ['person']
     mask = np.zeros((img['height'], img['width']))
     for i in range(len(anns)):
         className = getClassName(anns[i]['category_id'], cats)
         pixel_value = filterClasses.index(className) + 1
         mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)
     mask2 = cv2.convertScaleAbs(mask, alpha=(255.0))

     x = 'E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/train/CreatedMaskImages/image' + str(q) + '.jpg'
     cv2.imwrite(filename=x, img=mask2)
GetPhotos()
def LoadImages():
    ########## ALl POSSIBLE COMBINATIONS ########
    classes = ['person']

    images = []
    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given class
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    for q in range(len(imgIds)):
     img = coco.loadImgs(imgIds[q])[0]
     I = io.imread('{}/images/{}/{}'.format(dataDir, dataType, img['file_name'])) / 255.0
     img2 = cv2.convertScaleAbs(I, alpha=(255.0))
     x = 'E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/train/CreatedTrain/image' + str(q) + '.jpg'
     cv2.imwrite(filename=x, img=img2)
    dataset_size = len(unique_images)

    print("Number of images containing the filter classes:", dataset_size)


#LoadImages()
def LoadMasks(img):
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    filterClasses = ['person']
    mask = np.zeros((img['height'], img['width']))
    for i in range(len(anns)):
        className = getClassName(anns[i]['category_id'], cats)
        pixel_value = filterClasses.index(className) + 1
        mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)
    mask2 = cv2.convertScaleAbs(mask, alpha=(255.0))

    x = 'E:/ite/emotion_detection-master/COCOdataset2017/CreatedData/train/CreatedMaskImages/image' + str(q) + '.jpg'
    cv2.imwrite(filename=x, img=mask2)
