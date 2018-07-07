from __future__ import print_function
import numpy as np
import random
import cv2
import sys, os, re
import warnings
import argparse

random.seed(42)
np.random.seed(42) # for reproducibility

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


warnings.filterwarnings('ignore')

K.set_image_data_format('channels_last')

if K.backend() == 'tensorflow':
	import tensorflow as tf    # Memory control with Tensorflow
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	K.set_session(sess)


# ----------------------------------------------------------------------------
def list_files(directory, ext=None):
	return [os.path.join(directory, f) for f in os.listdir(directory)
			if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

# ----------------------------------------------------------------------------
def list_dirs(directory):
	return [os.path.join(directory, f) for f in os.listdir(directory)
			if os.path.isdir(os.path.join(directory, f))]

# ----------------------------------------------------------------------------
def load_images(path,digit,limit):
	X = []
	Y = []        
	label = str(digit)
	for fname in list_files( path, ext='png' ): 
		img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
		X.append(img)
		Y.append(label)

	combined = list(zip(X, Y))
	random.shuffle(combined)

	X[:], Y[:] = zip(*combined)

	if limit!=-1:
		X = X[:limit]
		Y = Y[:limit]

	X = np.asarray(X).astype('float32')
	X /= 255.
	X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

	nb_classes = 10
	Y = np_utils.to_categorical(Y, nb_classes)

	#print(Y)

	return X, Y

# ----------------------------------------------------------------------------
def get_cnn(input_shape, num_classes):
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('-path',    required=True,  type=str,   help='Dataset path')
parser.add_argument('-vae',     default=None,   type=str,   help='Path to VAE augmentation folder')
parser.add_argument('--aug',    action='store_true',        help='Use Keras augmentation')
parser.add_argument('-limtr',   default=100,    type=int,   help='Limit train size. -1 to load all images')
parser.add_argument('-limte',   default=125,    type=int,   help='Limit test size. -1 to load all images')
parser.add_argument('-e',       default=60,    type=int,   help='nb epochs')
parser.add_argument('-b',       default=32,     type=int,   help='batch size')
parser.add_argument('--v',      action='store_true',        help='Activate verbose')
parser.add_argument('--bad',    action='store_true',        help='Use BAD Keras augmentation')
args = parser.parse_args()

train_path = os.path.join(args.path, 'train\\')
test_path = os.path.join(args.path, 'test\\')

train_path_0 = os.path.join(train_path, '0')
test_path_0 = os.path.join(test_path, '0')

#print("*** MNIST ***")
#print("Loading digit 0")

X_train, Y_train = load_images(train_path_0, 0, args.limtr)
X_test, Y_test = load_images(test_path_0, 0, args.limte)

for digit in range(1, 10):
	#print("Loading digit %d" % (digit) )
	t_path = os.path.join(train_path, '%d'%(digit))
	X_train_aux, Y_train_aux = load_images(t_path, digit, args.limtr)
	X_train = np.concatenate((X_train, X_train_aux), axis=0)
	Y_train = np.concatenate((Y_train, Y_train_aux), axis=0)

	te_path = os.path.join(test_path, '%d'%(digit))
	X_test_aux, Y_test_aux = load_images(te_path, digit, args.limte)
	X_test = np.concatenate((X_test, X_test_aux), axis=0)
	Y_test = np.concatenate((Y_test, Y_test_aux), axis=0)

if args.vae != None:
	#print("\n*** VAE ***")
	#print("Loading digit 0")
	vae_path = os.path.join(args.vae, '0')
	X_vae, Y_vae = load_images(vae_path, 0, -1)
	for digit in range(1, 10):
		#print("Loading digit %d" % (digit) )
		vae_path = os.path.join(args.vae, '%d'%(digit))
		X_vae_aux, Y_vae_aux = load_images(vae_path, digit, -1)
		X_vae = np.concatenate((X_vae, X_vae_aux), axis=0)
		Y_vae = np.concatenate((Y_vae, Y_vae_aux), axis=0)
	
	X_vae, Y_vae = shuffle(X_vae, Y_vae, random_state=42)

print("\n########################################################################################")

X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
X_test, Y_test = shuffle(X_test, Y_test, random_state=42)

input_shape = X_train.shape[1:]

print('Path:', args.path)
print('Keras aug:', args.aug)
print('VAE aug:', args.vae)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
print('input_shape:', input_shape)

nb_classes = 10
model = get_cnn(input_shape, nb_classes)
#print(model.summary())


# Train

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

checkpoint = ModelCheckpoint('weights_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)

if (args.aug == True) and (args.vae != None):
	datagen = ImageDataGenerator(
					rotation_range=20,
					width_shift_range=0.2,
					height_shift_range=0.2,
					horizontal_flip=False)

	pagination_size = 50
	early_stopping_vae = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')
	checkpoint_vae = ModelCheckpoint('weights_best.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
	for i in range(0,args.e):
		#print("EPOCH %d" % ((i+1)))
		since 	= i * pagination_size
		until	= (i+1) * pagination_size

		X_aux = X_vae[since:until]
		Y_aux = Y_vae[since:until]

		X = np.concatenate((X_train, X_aux), axis=0)
		Y = np.concatenate((Y_train, Y_aux), axis=0)

		model.fit_generator(datagen.flow(X_train, Y_train, batch_size=args.b
						#, save_to_dir='AUGMENTATION'),
						),
				   steps_per_epoch=len(X_train), validation_data=(X_test, Y_test), 
				   epochs=1, 
				   verbose=0, 
				   callbacks=[checkpoint_vae])

elif args.bad == True:
	datagen = ImageDataGenerator(
                 zca_whitening=True,    # Si ves que tarda mucho con esto lo quitas
                 zca_epsilon=1e-6,
                 rotation_range=360,
                 width_shift_range=1.0,
                 height_shift_range=1.0,
                 shear_range=90.,   # revisa
                 zoom_range=5.,
                 fill_mode='nearest',
                 horizontal_flip=True,
                 vertical_flip=True)
	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=args.b
						#, save_to_dir='AUGMENTATION'),
						),
				   steps_per_epoch=len(X_train), validation_data=(X_test, Y_test), 
				   epochs=args.e, 
				   verbose=0, 
				   callbacks=[early_stopping, checkpoint])

elif args.aug == True:
	datagen = ImageDataGenerator(
					rotation_range=20,
					width_shift_range=0.2,
					height_shift_range=0.2,
					horizontal_flip=False)

	#datagen.fit(X_train)  # Not required

	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=args.b
						#, save_to_dir='AUGMENTATION'),
						),
				   steps_per_epoch=len(X_train), validation_data=(X_test, Y_test), 
				   epochs=args.e, 
				   verbose=0, 
				   callbacks=[early_stopping, checkpoint])

elif args.vae != None:
	pagination_size = 100
	early_stopping_vae = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')
	checkpoint_vae = ModelCheckpoint('weights_best.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
	for i in range(0,args.e):
		#print("EPOCH %d" % ((i+1)))
		since 	= i * pagination_size
		until	= (i+1) * pagination_size

		X_aux = X_vae[since:until]
		Y_aux = Y_vae[since:until]

		X = np.concatenate((X_train, X_aux), axis=0)
		Y = np.concatenate((Y_train, Y_aux), axis=0)

		model.fit(X,Y,epochs=1,
			batch_size=args.b,
			verbose=0,
			validation_data=(X_test, Y_test),
			callbacks=[checkpoint_vae])

else:
	model.fit(X_train, Y_train,
		  batch_size=args.b,
		  epochs=args.e,
		  verbose=0, 
		  validation_data=(X_test, Y_test),
		  callbacks=[early_stopping, checkpoint])


# Evaluate
model.load_weights('weights_best.h5')

score = model.evaluate(X_test, Y_test) #, verbose==1 if args.v == True else 2)

Y_pred = model.predict(X_test, batch_size=args.b, verbose=0)
Y_pred1 = np.argmax(Y_pred, axis=1)
Y_test1 = list()
for i in Y_test:
	Y_test1.append(np.argmax(i))

accuracy = accuracy_score(Y_test1, Y_pred1)    
precision, recall, f1, support = precision_recall_fscore_support(Y_test1, Y_pred1, average=None)

print('\nFinal results...')
print(classification_report(Y_test1, Y_pred1))

print('Loss     : %.4f' % score[0])
print('Acc      : %.4f' % accuracy)
print('Precision: %.4f' % np.average(precision))
print('Recall   : %.4f' % np.average(recall))
print('F1       : %.4f' % np.average(f1))
print('Support  :', np.sum(support))

