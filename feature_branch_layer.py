import caffe

import numpy as np
from PIL import Image
import random
import scipy.io

class FaceRecogDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """
	# def __init__(self):
    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - faceImage_dir: path to face image dir
        - batch: batch_size
        - mean: tuple of mean values to subtract
        - label_dir: directory to stor label information, .mat format
		- split(non): 
        """
        # config
        params = eval(self.param_str)
        self.faceImage_dir = params['faceImage_dir']

        # three tops: data, label and landmark
        if len(top) != 3:
            raise Exception("Need to define three tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
		
		
		# load indices labels
        split_f  = '{}/{}.txt'.format(self.dataLabel_dir,
                'train')
        self.indices = open(split_f, 'r').read().splitlines()
        print self.indices
        print len(self.indices)
        print self.totalImage
		
        self.data0 = self.load_image(self.indices[1])
        print self.data0.shape
        self.data = np.zeros(shape=(self.batch, 3, 227, 227))
        print self.data.shape
        self.label = np.zeros(shape=(self.batch))
        print self.label.shape
		
		# build the feature label vector
        self.landMark0 = np.zeros(shape=(128, 2))
        self.landMark = np.zeros(shape=(self.batch, 128, 2))

		
        for num in range(0, self.batch):
            print self.batch
            print num
            self.idx = random.randint(1, self.totalImage)
            self.data[num] = self.load_image(self.idx)
            self.label[num] = self.indices[self.idx-1]
            self.landMark[num] = self.load_feature(self.idx)

        print self.data.shape
        self.data = np.array(self.data, dtype=np.float32)
        self.label = np.array(self.label, dtype=np.float32)
        self.landMark = np.array(self.landMark, dtype=np.float32)
        print self.data.shape
		
    def forward(self, bottom, top):
        # assign output
        # self.reshape(self.bottom, self.top)
        print "forward"
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.landMark
        print "landmark"


    def backward(self, top, propagate_down, bottom):
        print "backward"
		
    def reshape(self, bottom, top):
        print "reshape"	
        top[0].reshape(self.batch, *self.data0.shape)
        top[1].reshape(self.batch)
        top[2].reshape(self.batch, *self.landMark0.shape)
		
    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #self.faceImage_dir = params['faceImage_dir']
        im = Image.open('{}/{}.jpg'.format(self.faceImage_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_
	
    def load_feature(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        print idx
        mat = scipy.io.loadmat('{}/{}.mat'.format(self.landMark_dir, idx))
        print "matInput check"
        print mat
        print mat['data']
        matArray = np.array(mat['data'], dtype=np.float32)
        # label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        # label = label[np.newaxis, ...]
        matArray.flatten()
        return matArray	