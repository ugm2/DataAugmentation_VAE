import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import pickle
from PIL import Image
import cv2

import sys

from features import *

if len(sys.argv)>3:
    epochs = int(sys.argv[3])

if len(sys.argv)>4:
  latent_dim = int(sys.argv[4])

# tensorflow or theano
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

# encoder architecture
x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(int(filters * img_rows / 2 * img_cols / 2), activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, int(img_rows / 2), int(img_cols / 2))
else:
    output_shape = (batch_size, int(img_rows / 2), int(img_cols / 2), filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
# if K.image_data_format() == 'channels_first':
#     output_shape = (batch_size, filters, 29, 29)
# else:
#     output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
#print(vae.summary())


# build a model to project inputs on the latent space
encoder = Model(x, [z_mean, z_log_var, z])


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)



label = sys.argv[1]

if len(sys.argv)>2:
    TOTAL_IMAGES = int(sys.argv[2])

# load saved models
vae.load_weights('../models/%d/%s/mnist_ld_%d_conv_%d_id_%d_e_%d_vae.h5' % (TOTAL_IMAGES, label, latent_dim, num_conv, intermediate_dim, epochs)
    #,custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer}
    )
encoder.load_weights('../models/%d/%s/mnist_ld_%d_conv_%d_id_%d_e_%d_encoder.h5' % (TOTAL_IMAGES, label,latent_dim, num_conv, intermediate_dim, epochs)
    #,custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer}
    )
generator.load_weights('../models/%d/%s/mnist_ld_%d_conv_%d_id_%d_e_%d_generator.h5' % (TOTAL_IMAGES, label,latent_dim, num_conv, intermediate_dim, epochs)
    #,custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer}
    )


# GENERATE ALL IMAGES (2500 EACH CLASS) #
n = 50
img_size = 28

directory = "../datasets/generated/%d_dim/%d/%s_2/" % (latent_dim, TOTAL_IMAGES, label)

if not os.path.exists(directory):
        os.makedirs(directory)
'''
imgCount = 0
for i in range(n):
    for j in range(n):
        imgCount+=1
        z_sample = np.array([np.random.uniform(-1.5,1.5 ,size=latent_dim)])
        x_decoded = generator.predict(z_sample)
        img = x_decoded[0].reshape(28, 28, img_chns)
        img2 = img*255
        cv2.imwrite('%s%d.png' % (directory, imgCount), img2)
'''
imgCount = 0
for i in range(n):
    for j in range(n):
        imgCount+=1
        z_sample = np.array([np.random.uniform(-1.5,1.5 ,size=latent_dim)])
        x_decoded = generator.predict(z_sample)
        img = x_decoded[0].reshape(28, 28, img_chns)
        img2 = img*255
        cv2.imwrite('%s%d.png' % (directory, imgCount), img2)

'''
# GENERATE ONE RANDOM IMAGE #
directory = "images/%s/" % (TOTAL_IMAGES)

if not os.path.exists(directory):
  os.makedirs(directory)

img_size = 28
z_sample = np.array([np.random.uniform(-1.5, 1.5,size=latent_dim)])
x_decoded = generator.predict(z_sample)
dig = x_decoded[0].reshape(img_size, img_size)
dig *= 255
dig = 255 - dig
cv2.imwrite(("%s/%s.png" % (directory,label)), dig)

# display images generated from randomly sampled latent vector
'''

'''
# RESEARCH ON ANOTHER WAY TO GENERATE THE IMAGES #
def load_images(path):
    X = []
    Y = []        
    for fname in list_files( path, ext='png' ): 
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        X.append(img)
        Y.append(label)

    combined = list(zip(X, Y))
    random.shuffle(combined)

    X[:], Y[:] = zip(*combined)

    X = X[:TOTAL_IMAGES]
    Y = Y[:TOTAL_IMAGES]

    X = np.asarray(X).astype('float32')
    X = X/255.
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    Y = np.asarray(Y)
    
    return X, Y

def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]


X, Y = load_images("../datasets/train/%s" %(label))

if (X.shape[0]<10):
    iter_size = X.shape[0]
else:
    iter_size = 10

for iterations in range(iter_size):

    directory = "../imagenes_prueba/other_way/%s/%d_%d/%d/" %(label, latent_dim, TOTAL_IMAGES, iterations)

    if not os.path.exists(directory):
            os.makedirs(directory)

    sample = X[iterations]
    sampleCV2 = sample*255
    cv2.imwrite('%sORIGINAL.png' % (directory), sampleCV2)
    sample = np.expand_dims(sample, axis=0)

    imgCount = 0
    for it in range(30):
        imgCount+=1
        mean, sigma, _ = encoder.predict(sample)
        print(mean)
        print(sigma)
        epsilon = np.array([np.random.uniform(-1.5+(0.1*it),-1.5+(0.1*(it+1)) ,size=latent_dim)])
        encoding_part = mean + K.exp(sigma) * epsilon
        print(encoding_part)
        x_decoded = generator.predict(encoding_part)
        img = x_decoded[0].reshape(28, 28, img_chns)
        img2 = img*255
        cv2.imwrite('%s%d.png' % (directory, imgCount), img2)

'''