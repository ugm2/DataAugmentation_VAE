from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics

import sys

from features import *

if len(sys.argv)>3:
  epochs = int(sys.argv[3])

if len(sys.argv)>4:
  latent_dim = int(sys.argv[4])

if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

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
        return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
#print(vae.summary())


label = int(sys.argv[1])

#### DATA ####
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

    Y = np.asarray(Y)
    
    return X, Y

def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

if len(sys.argv)>2:
    TOTAL_IMAGES = int(sys.argv[2])

directory = "../datasets/train/%s/" %(label)

X, y = load_images(directory)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.16)

print("LATENT DIMENSION: ", latent_dim)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

print('x_train.shape:', x_train.shape)

history = vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        verbose=0)

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


# save all 3 models for future use - especially generator
vae.save('../models/%d/%s/mnist_ld_%d_conv_%d_id_%d_e_%d_vae.h5' % (TOTAL_IMAGES, label, latent_dim, num_conv, intermediate_dim, epochs))
encoder.save('../models/%d/%s/mnist_ld_%d_conv_%d_id_%d_e_%d_encoder.h5' % (TOTAL_IMAGES, label, latent_dim, num_conv, intermediate_dim, epochs))
generator.save('../models/%d/%s/mnist_ld_%d_conv_%d_id_%d_e_%d_generator.h5' % (TOTAL_IMAGES, label, latent_dim, num_conv, intermediate_dim, epochs))

n = 3
img_size = 28
figure = np.zeros((img_size * n, img_size * n))

for i in range(n):
    for j in range(n):
        z_sample = np.array([np.random.uniform(-1.5, 1.5,size=latent_dim)])
        x_decoded = generator.predict(z_sample)
        dig = x_decoded[0].reshape(img_size, img_size)
        figure[i * img_size: (i + 1) * img_size,j * img_size: (j + 1) * img_size] = dig

plt.figure(figsize=(20, 20))
plt.imshow(figure, cmap='Greys_r')
#plt.show()