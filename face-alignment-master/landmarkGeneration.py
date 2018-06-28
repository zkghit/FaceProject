import argparse
import face_alignment
from skimage import io
# from skimage.transform import resize
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type = bool, default=True)
parser.add_argument('--load_dir', type = str, default='../Facephoto/')
opt = parser.parse_args()

if __name__ == "__main__":
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False)
	for file in os.listdir(opt.load_dir):
		if file.endswith(".jpg"):
			input = io.imread(opt.load_dir+file)
			# imResize = resize(input,(200,200))
			preds = fa.get_landmarks(input)
			name = file.replace('.jpg','.npy')
			np.save(name, preds[0])
			nameJ = file.replace('.npy','.jpg')
			io.imsave(nameJ, input)

# sio.savemat('small.mat',{preds})
# data = np.load('test.npy')
