import json
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from math import ceil, sqrt
import random
import os

class pre(object):
	def __init__(self, path, decimals=1):
		self.smiles = []
		self.propertys = []
		self.path = path
		self.mols = []
		self.train_mols = []
		self.train_pro = []
		self.train_pre02 = []
		self.train_pre23 = []
		self.train_pre34 = []
		self.train_graph = []
		self.test_mols = []
		self.test_pro = []
		self.test_pre02 = []
		self.test_pre23 = []
		self.test_pre34 = []
		self.test_graph = []
		self.distance = [0, 0, 0]
		self.decimals = decimals
		self.multiple = 10 ** self.decimals
		self.ai = {"H": 0, "He": 1, "Li": 2, "Be": 3, "B": 4, "C": 5, "N": 6, "O": 7, "F": 8, "Ne": 9, "Na": 10, "Mg": 11, "Al": 12, "Si": 13, "P": 14, "S": 15, "Cl": 16, "Ar": 17, "K": 18, "Ca": 19, "Sc": 20, "Ti": 21, "V": 22, "Cr": 23, "Mn": 24, "Fe": 25, "Co": 26, "Ni": 27, "Cu": 28, "Zn": 29, "Ga": 30, "Ge": 31, "As": 32, "Se": 33, "Br": 34, "Kr": 35, "Rb": 36, "Sr": 37, "Y": 38, "Zr": 39, "Nb": 40, "Mo": 41, "Tc": 42, "Ru": 43, "Rh": 44, "Pd": 45, "Ag": 46, "Cd": 47, "In": 48, "Sn": 49, "Sb": 50, "Te": 51, "I": 52, "Xe": 53, "Cs": 54, "Ba": 55, "La": 56, "Ce": 57, "Pr": 58, "Nd": 59, "Pm": 60, "Sm": 61, "Eu": 62, "Gd": 63, "Tb": 64, "Dy": 65, "Ho": 66, "Er": 67, "Tm": 68, "Yb": 69, "Lu": 70, "Hf": 71, "Ta": 72, "W": 73, "Re": 74, "Os": 75, "Ir": 76, "Pt": 77, "Au": 78, "Hg": 79, "Tl": 80, "Pb": 81, "Bi": 82, "Po": 83, "At": 84, "Rn": 85, "Fr": 86, "Ra": 87, "Ac": 88, "Th": 89, "Pa": 90, "U": 91, "Np": 92, "Pu": 93, "Am": 94, "Cm": 95, "Bk": 96, "Cf": 97, "Es": 98, "Fm": 99, "Md": 100, "No": 101, "Lr": 102, "Rf": 103, "Db": 104, "Sg": 105, "Bh": 106, "Hs": 107, "Mt": 108, "Ds": 109, "Rg": 110, "Cn": 111}


	def load_data(self):
		with open(self.path + 'data/dsgdb9nsd_finput.xyz', 'r') as f:
			self.mols = json.load(f)
		with open(self.path + 'data/dsgdb9nsd_label.xyz', 'r') as f:
			self.propertys = json.load(f)

	def longest_distance(self):
		for i in range(len(self.mols)):
			coor_xyz = []
			mol = self.mols[i]
			for j in range(len(mol)):
				_coor_xyz = []
				for k in [1,2,3]:
					try:
						_coor_xyz.append(float(mol[j][k]))
					except:
						_coor_xyz.append(0)
				coor_xyz.append(_coor_xyz)
			coor_xyz = np.array(coor_xyz)
			min_x = min(coor_xyz[:,0])
			min_y = min(coor_xyz[:,1])
			min_z = min(coor_xyz[:,2])
			max_x = max(coor_xyz[:,0])
			max_y = max(coor_xyz[:,1])
			max_z = max(coor_xyz[:,2])
			if abs(max_x - min_x) > self.distance[0]:
				self.distance[0] = abs(max_x - min_x)
			if abs(max_y - min_y) > self.distance[1]:
				self.distance[1] = abs(max_y - min_y)
			if abs(max_z - min_z) > self.distance[2]:
				self.distance[2] = abs(max_z - min_z)
		with open(self.path + 'data/distance.json', 'w') as f:
			json.dump(self.distance, f)
		with open(self.path + 'data/information.txt', 'a') as f:
			f.write('Distance:\nX:%f, Y%f, Z%f' %(self.distance[0], self.distance[1], self.distance[2]))

	def correction(self):
		mol = []
		pppp = []
		ori_mol = self.mols
		ori_pro = []
		print(len(self.mols))
		print(len(self.propertys))
		for i in range(len(self.mols)):
			coor_xyz = []
			mol = self.mols[i]
			for j in range(len(mol)):
				_coor_xyz = []
				for k in [1,2,3]:
					try:
						_coor_xyz.append(float(mol[j][k]))
					except:
						_coor_xyz.append(0)
				coor_xyz.append(_coor_xyz)
			coor_xyz = np.array(coor_xyz)
			min_x = min(coor_xyz[:,0])
			min_y = min(coor_xyz[:,1])
			min_z = min(coor_xyz[:,2])
			max_x = max(coor_xyz[:,0])
			max_y = max(coor_xyz[:,1])
			max_z = max(coor_xyz[:,2])
			coor_xyz = coor_xyz.tolist()
			dx = max_x - min_x
			dy = max_y - min_y
			dz = max_z - min_z
			if dx > 12 or dy > 12 or dz > 12:
				continue
			for j in range(len(mol)):
				coor_xyz[j][0] -= (min_x + 0.5 * dx - 0.5 * 12)
				coor_xyz[j][1] -= (min_y + 0.5 * dy - 0.5 * 12)
				coor_xyz[j][2] -= (min_z + 0.5 * dz - 0.5 * 12)
			ppp = []
			pp = []
			for ppc in range(len(coor_xyz)):
				pp = [mol[ppc][0]] + coor_xyz[ppc]
				ppp.append(pp)
			pppp.append(ppp)
			ori_pro.append(self.propertys[i])
		self.mols = pppp
		self.propertys = ori_pro
		with open(self.path + 'data/all_coor_corr.json', 'w') as f:
			json.dump(self.mols, f)
		with open(self.path + 'data/all_pro_corr.json', 'w') as f:
			json.dump(self.propertys, f)

	def shuffle(self):
		index = np.arange(len(self.mols))
		random.shuffle(index)
		X = [self.mols[int(i)] for i in index]
		Y = [self.propertys[int(i)] for i in index]
		self.train_mols = X[:int(len(X)*0.9)]
		self.train_pro = Y[:int(len(Y)*0.9)]
		self.test_mols = X[int(len(X)*0.9):]
		self.test_pro = Y[int(len(Y)*0.9):]
		with open(self.path + 'data/information.txt', 'a') as f:
			f.write('Molecule numbers: \nTrain:%d, test:%d \n' %(len(self.train_mols), len(self.test_mols)))
		with open(self.path + 'data/train_mols.json', 'w') as f:
			json.dump(self.train_mols, f)
		with open(self.path + 'data/train_pro.json', 'w') as f:
			json.dump(self.train_pro, f)
		with open(self.path + 'data/test_mols.json', 'w') as f:
			json.dump(self.test_mols, f)
		with open(self.path + 'data/test_pro.json', 'w') as f:
			json.dump(self.test_pro, f)

	def computepixels(self, atomi, xi, yi, zi, atomj, xj, yj, zj):
		if atomi == 'H' or atomj == 'H':
			return [[],[],[]]
		temstorage02 = set()
		temstorage23 = set()
		temstorage34 = set()
		storage02 = []
		storage23 = []
		storage34 = []
		try:
			D = round(1 / (sqrt((xi - xj) ** 2 + (yi - yj) ** 2)), 4)
		except:
			return [[],[],[]]
		if D >= (1/2):
			xi = int(100 * xi)
			yi = int(100 * yi)
			zi = int(100 * zi)
			xj = int(100 * xj)
			yj = int(100 * yj)
			zj = int(100 * zj)
			Dx = abs(xi - xj)
			kxy = np.polyfit([xi, xj], [yi, yj], 1)
			kxz = np.polyfit([xi, xj], [zi, zj], 1)
			x = min(xi, xj)
			for dx in range(Dx - 1):
				x_ = int(ceil((x + dx) / 100 * self.multiple / 2))
				y_ = int(ceil((kxy[0] * (x + dx) + kxy[1]) / 100 * self.multiple / 2))
				z_ = int(ceil((kxz[0] * (x + dx) + kxz[1]) / 100 * self.multiple / 2))
				temstorage02.add((x_, y_, z_, self.ai[atomi], self.ai[atomj], D))
				dx += 1
		elif D >= (1/3) and D < (1/2):
			xi = int(100 * xi)
			yi = int(100 * yi)
			zi = int(100 * zi)
			xj = int(100 * xj)
			yj = int(100 * yj)
			zj = int(100 * zj)
			Dx = abs(xi - xj)
			kxy = np.polyfit([xi, xj], [yi, yj], 1)
			kxz = np.polyfit([xi, xj], [zi, zj], 1)
			x = min(xi, xj)
			for dx in range(Dx - 1):
				x_ = int(ceil((x + dx) / 100 * self.multiple / 2))
				y_ = int(ceil((kxy[0] * (x + dx) + kxy[1]) / 100 * self.multiple / 2))
				z_ = int(ceil((kxz[0] * (x + dx) + kxz[1]) / 100 * self.multiple / 2))
				temstorage23.add((x_, y_, z_, self.ai[atomi], self.ai[atomj], D))
				dx += 1
		elif D >= (1/4) and D < (1/3):
			xi = int(100 * xi)
			yi = int(100 * yi)
			zi = int(100 * zi)
			xj = int(100 * xj)
			yj = int(100 * yj)
			zj = int(100 * zj)
			Dx = abs(xi - xj)
			kxy = np.polyfit([xi, xj], [yi, yj], 1)
			kxz = np.polyfit([xi, xj], [zi, zj], 1)
			x = min(xi, xj)
			for dx in range(Dx - 1):
				x_ = int(ceil((x + dx) / 100 * self.multiple / 2))
				y_ = int(ceil((kxy[0] * (x + dx) + kxy[1]) / 100 * self.multiple / 2))
				z_ = int(ceil((kxz[0] * (x + dx) + kxz[1]) / 100 * self.multiple / 2))
				temstorage34.add((x_, y_, z_, self.ai[atomi], self.ai[atomj], D))
				dx += 1
		else:
			return [[],[],[]]		
		for i in temstorage02:
			storage02.append(list(i))
		for i in temstorage23:
			storage23.append(list(i))
		for i in temstorage34:
			storage34.append(list(i))
		return [storage02, storage23, storage34]

	def image_pre(self, _train=True, _test=True):
		if _train == True:
			for index in range(len(self.train_mols)):
				temstorage02 = []
				temstorage23 = []
				temstorage34 = []
				_mol = self.train_mols[index]
				for i in range(len(_mol)):
					_atom = _mol[i]
					atomi = _atom[0]
					xi = _atom[1]
					yi = _atom[2]
					zi = _atom[3]
					for j in range(i+1, len(_mol)):
						__atom = _mol[j]
						atomj = __atom[0]
						xj = __atom[1]
						yj = __atom[2]
						zj = __atom[3]
						pixel = self.computepixels(atomi, xi, yi, zi, atomj, xj, yj, zj)
						if pixel[0] != []:
							temstorage02.extend(pixel[0])
						elif pixel[1] != []:
							temstorage23.extend(pixel[1])
						elif pixel[2] != []:
							temstorage34.extend(pixel[2])
						else:
							continue
				self.train_pre02.append(temstorage02)
				self.train_pre23.append(temstorage23)
				self.train_pre34.append(temstorage34)
			with open(self.path + 'data/train_pre02.json', 'w') as f:
				json.dump(self.train_pre02, f)
			with open(self.path + 'data/train_pre23.json', 'w') as f:
				json.dump(self.train_pre23, f)
			with open(self.path + 'data/train_pre34.json', 'w') as f:
				json.dump(self.train_pre34, f)
		if _test == True:
			for index in range(len(self.test_mols)):
				temstorage02 = []
				temstorage23 = []
				temstorage34 = []
				_mol = self.test_mols[index]
				for i in range(len(_mol)):
					_atom = _mol[i]
					atomi = _atom[0]
					xi = _atom[1]
					yi = _atom[2]
					zi = _atom[3]
					for j in range(i+1, len(_mol)):
						__atom = _mol[j]
						atomj = __atom[0]
						xj = __atom[1]
						yj = __atom[2]
						zj = __atom[3]
						pixel = self.computepixels(atomi, xi, yi, atomj, xj, yj)
						if pixel[0] != []:
							temstorage02.extend(pixel[0])
						elif pixel[1] != []:
							temstorage23.extend(pixel[1])
						elif pixel[2] != []:
							temstorage34.extend(pixel[2])
						else:
							continue
				self.test_pre02.append(temstorage02)
				self.test_pre23.append(temstorage23)
				self.test_pre34.append(temstorage34)
			with open(self.path + 'data/test_pre02.json', 'w') as f:
				json.dump(self.test_pre02, f)
			with open(self.path + 'data/test_pre23.json', 'w') as f:
				json.dump(self.test_pre23, f)
			with open(self.path + 'data/test_pre34.json', 'w') as f:
				json.dump(self.test_pre34, f)


def pre_main(path='/public/home/hwj/zh/local-connect/MIC/'):
	main = pre(path)
	main.load_data()
	main.longest_distance()
	main.correction()
	main.shuffle()
	main.image_pre()

from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Dropout,Input 
from keras.layers.convolutional import Conv3D,MaxPooling3D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers.normalization import BatchNormalization as BN
from keras.utils import multi_gpu_model 
import numpy as np
import h5py 
import os
import zipfile 
import json
import time
from math import ceil, floor
from keras import layers
from random import random
from math import pi,cos,sin

# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'  

class trainer(object):
	def __init__(self, path, lr=0.001, batch_size=128, epochs=10000, out=1, _type='classify', class_weight=None):
		self.lr= lr
		self.path = path
		self.batch_size = batch_size
		self.epochs = epochs
		self.out = out
		self.type = _type
		self.model = None
		self.val_graph = None
		self.val_pro = None
		self.train_graph = None
		self.train_pro = None
		self.class_weight = class_weight
		if self.type == 'classify':
			if self.out == 1:
				self.loss = 'binary_crossentropy'
			else:
				self.loss = 'categorical_crossentropy'
		else:
			self.loss = 'mean_squared_error'

	def load_data(self):
		with open(self.path + 'data/train_graph.json', 'r') as f:
			self.train_graph = json.load(f)
		with open(self.path + 'data/train_pro.json', 'r') as f:
			self.train_pro = json.load(f)
		with open(self.path + 'data/val_graph.json', 'r') as f:
			self.val_graph = json.load(f)
		with open(self.path + 'data/val_pro.json', 'r') as f:
			self.val_pro = json.load(f)

	def model_build(self):
		inputs_02 = Input(shape=((96,96,96,3)))
		x1 = Conv3D(32,(1,1,1),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(inputs_02)
		x1 = BN()(x1)
		x11 = Conv3D(48,(11,11,11),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x1)
		x11 = BN()(x11)
		x12 = Conv3D(48,(7,7,7),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x1)
		x12 = BN()(x12)
		x13 = Conv3D(48,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x1)
		x13 = BN()(x13)
		x1 = layers.concatenate([x11, x12, x13], axis=-1)
		x1 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x1) # 192 -> 96 / 128 -> 64
		x1 = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x1)

		inputs_23 = Input(shape=((96,96,96,3)))
		x2 = Conv3D(32,(1,1,1),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(inputs_23)
		x2 = BN()(x2)
		x21 = Conv3D(48,(11,11,11),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x2)
		x21 = BN()(x21)
		x22 = Conv3D(48,(7,7,7),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x2)
		x22 = BN()(x22)
		x23 = Conv3D(48,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x2)
		x23 = BN()(x23)
		x2 = layers.concatenate([x21, x22, x23], axis=-1)
		x2 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x2) # 192 -> 96 / 128 -> 64
		x2 = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x2)

		inputs_34 = Input(shape=((96,96,96,3)))
		x3 = Conv3D(32,(1,1,1),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(inputs_34)
		x3 = BN()(x3)
		x31 = Conv3D(48,(11,11,11),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x3)
		x31 = BN()(x31)
		x32 = Conv3D(48,(7,7,7),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x3)
		x32 = BN()(x32)
		x33 = Conv3D(48,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x3)
		x33 = BN()(x33)
		x3 = layers.concatenate([x31, x32, x33], axis=-1)
		x3 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x3) # 192 -> 96 / 128 -> 64
		x3 = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x3)

		x0 = layers.concatenate([x1,x2,x3], axis=-1)
		x0 = Conv3D(128,(1,1,1),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x0)
		x01 = Conv3D(96,(7,7,7),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x0)
		x01 = BN()(x01)
		x02 = Conv3D(96,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x0)
		x02 = BN()(x02)
		x0 = layers.concatenate([x01, x02], axis=-1)
		x0 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x0) # 96 -> 48 / 64 -> 32

		x01 = Conv3D(96,(7,7,7),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x0)
		x01 = BN()(x01)
		x02 = Conv3D(96,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x0)
		x02 = BN()(x02)
		x0 = layers.concatenate([x01, x02], axis=-1)
		x0 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x0) # 48 -> 24 / 32 -> 16

		x0 = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x0)
		x0 = BN()(x0)
		x0 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x0) # 24 -> 12 / 16 -> 8

		x0 = Conv3D(384,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x0)
		x0 = BN()(x0)
		x0 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x0) # 12 -> 6 / 8 -> 4

		x0 = Conv3D(512,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_initializer='uniform')(x0)
		x0 = BN()(x0)
		x0 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x0) # 6 -> 3/ 4 -> 2

		x0 = Flatten()(x0)
		x0 = Dense(1024,activation='relu')(x0)
		x0 = BN()(x0)
		x0 = Dense(2048,activation='relu')(x0)
		x0 = BN()(x0)
		predictions = Dense(self.out)(x0)
		self.model = Model(inputs = [inputs_02, inputs_23, inputs_34], outputs = predictions)
		self.model = multi_gpu_model(self.model, 8)
		self.model.compile(loss=self.loss, optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])


	def train(self):
		cnt = 1
		for i in range(self.epochs):
			hist = self.model.fit_generator(generator=self.spin(),
				steps_per_epoch=941,
				epochs=1)
			self.model.save_weights(self.path + 'log/model_weights' + str(i) + '.h5')
			with open(self.path + 'log/note.json', 'a') as f:
				json.dump(hist.history, f)
				f.write('\n')
			if (i+1) / 100 == 0:
				self.lr /= 1.5
				self.model.compile(loss=self.loss, optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])
				cnt += 1
				self.model.load_weights(self.path + 'log/model_weights' + str(i) + '.h5')

	def val(self):
		self.model_build()
		with open(self.path + 'logs/note.json') as f:
			note = f.readlines()
			all_epochs = len(note)
		with open(self.path + 'data/val_graph.json') as f:
			x = np.asarray(json.load(f))
		with open(self.path + 'data/val_pro.json') as f:
			y = np.asarray(json.load(f))
		for i in range(all_epochs):
			self.model.load_weights(self.path + 'logs/model_weights%d.h5' % (i))
			print(self.model.evaluate(x, y, batch_size = self.batch_size))

	def generator(self):
		cnt = 0
		with open(self.path + 'data/train_pro.json') as f:
			y = json.load(f)
			X = []
			Y = []
		while 1:
			with open(self.path + 'true_graph/train/%d_02.json' %(cnt)) as f:
				X_02 = json.load(f)
			with open(self.path + 'true_graph/train/%d_23.json' %(cnt)) as f:
				X_23 = (json.load(f))
			with open(self.path + 'true_graph/train/%d_34.json' %(cnt)) as f:
				X_34 = (json.load(f))
			X.append(X_02)
			X.append(X_23)
			X.append(X_34)
			Y.append(y[cnt])
			cnt += 1
			if cnt % self.batch_size == 0:
				yield (np.asarray(X), np.asarray(Y))
				X = []
				Y = []

	def val_generator(self):
		cnt = 0
		with open(self.path + 'data/val_pro.json') as f:
			y = json.load(f)
		while 1:
			X = []
			Y = []
			with open(self.path + 'true_graph/val/%d.json' %(cnt)) as f:
				X.append(json.load(f))
			Y.append(y[cnt])
			cnt += 1
			if cnt % self.batch_size == 0:
				yield (np.asarray(X), np.asarray(Y))

	def spin(self):
		with open(self.path + 'data/train_pre02.json', 'r') as f:
			train_pre02 = json.load(f)
		with open(self.path + 'data/train_pre23.json', 'r') as f:
			train_pre23 = json.load(f)
		with open(self.path + 'data/train_pre34.json', 'r') as f:
			train_pre34 = json.load(f)
		with open(self.path + 'data/train_pro.json', 'r') as f:
			train_pro = np.asarray(json.load(f))[:,11].tolist()
		for i in range(len(train_pro)):
			train_pro[i] = float(train_pro[i])
		for index in range(941):
			X_train02 = train_pre02[index*64:(index+1)*64]
			X_train23 = train_pre23[index*64:(index+1)*64]
			X_train34 = train_pre34[index*64:(index+1)*64]
			Y = train_pro[index*64:(index+1)*64]
			X02 = np.zeros(shape=(64,96,96,96,3))
			X23 = np.zeros(shape=(64,96,96,96,3))
			X34 = np.zeros(shape=(64,96,96,96,3))
			for i in range(64):
				sigma = random() * 2 * pi
				beta = random() * 2 * pi
				delta = random() * 2 * pi
				mol02 = X_train02[i]
				mol23 = X_train23[i]
				mol34 = X_train34[i]
				Rx = np.array([[1, 0, 0], [0, cos(sigma), -sin(sigma)], [0, sin(sigma), cos(sigma)]])
				Ry = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
				Rz = np.array([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])
				for j in range(len(mol02)):
					ato02 = mol02[j]
					coordinate = np.asarray([ato02[:3]]) - 60
					spin_coordinate = (np.dot(np.dot(np.dot(coordinate, Rx), Ry), Rz) + 60).tolist()
					X02[i][spin_coordinate[0]][spin_coordinate[1]][spin_coordinate[2]] = np.asarray(ato02[3:])
				for j in range(len(mol23)):
					ato23 = mol23[j]
					coordinate = np.asarray([ato23[:3]]) - 60
					spin_coordinate = (np.dot(np.dot(np.dot(coordinate, Rx), Ry), Rz) + 60).tolist()
					X23[i][spin_coordinate[0]][spin_coordinate[1]][spin_coordinate[2]] = np.asarray(ato23[3:])
				for j in range(len(mol34)):
					ato34 = mol34[j]
					coordinate = np.asarray([ato34[:3]]) - 60
					spin_coordinate = (np.dot(np.dot(np.dot(coordinate, Rx), Ry), Rz) + 60).tolist()
					X34[i][spin_coordinate[0]][spin_coordinate[1]][spin_coordinate[2]] = np.asarray(ato34[3:])
			yield ([X02, X23, X34], np.asarray(Y)[:,10])


	def main_trainer(self):

		self.model_build()
		self.train()

if __name__ == '__main__':
	pre_main(path='/public/home/hwj/zh/local-connect/MIC/')
	#main = trainer(path='/public/home/hwj/zh/multi_QM9/', lr=0.001, batch_size=128, epochs=1000, out=1, _type='classify', class_weight=None)
	#main.main_trainer()