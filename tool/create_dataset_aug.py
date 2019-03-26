import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import synthTransformer as synthTrans
from PIL import Image
from StringIO import StringIO
from multiprocessing import Pool
#Can get atmost 100 million unique images

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    #print(imgH, imgW)
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def createDataset(outputPath, imagePathList, labelList, parentDirofImages, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    p = Pool(8)
    no_aug = 20
    elastic = synthTrans.ElasticTransformation(0.5)
    affine  = synthTrans.AffineTransformation(0.5)

    imagePathList = open(imagePathList,'r').readlines()
    for i in range(0,len(imagePathList)):
    	imagePathList[i] = parentDirofImages +  imagePathList[i].strip() #+ '.png'
    labelList = open(labelList,'r').readlines()
    for i in range(0,len(labelList)):
     	labelList[i] = labelList[i].strip()
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        im_list = []
        im_list.extend([img]*no_aug)
        nva = p.map(affine, im_list)
        nva = p.map(elastic, nva)
        for k in xrange(0,len(nva)):
            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[labelKey] = label
            im = Image.fromarray(np.uint8(nva[k]))
            output = StringIO()
            contents = output.getvalue()
            output.close()
            cache[imageKey] = contents
            cnt+=1
            if lexiconList:
                lexiconKey = 'lexicon-%09d' % cnt
                cache[lexiconKey] = ' '.join(lexiconList[i])
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples*no_aug))
        
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

if __name__ == '__main__':
    createDataset('train-elastic-lmdb', '/data4/kartik/labpc_save/iam-list/train-img.txt', '/data4/kartik/labpc_save/iam-list/train-label.txt', '', None, True)
