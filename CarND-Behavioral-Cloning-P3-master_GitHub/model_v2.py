import cv2
import keras.models
import keras.layers
import keras.layers.core
import keras.layers.convolutional
import os
import sklearn
import sklearn.model_selection
import sklearn.utils
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# get python file name to name output .h5
fname = os.path.basename(sys.argv[0])
fname = os.path.splitext(os.path.basename(fname))[0]

# augment - Left and Right camera images with correction
augment_cameras_LR = True
augment_cameras_LR_factor = 0.08

# augment - flip images
augment_flip_left2right = False

# switch to train (or just process)
Train = True

# switch to enable CNN dropout - to prevent overfit
dropout = True

# image folders, description, on/off
dataf = ['data00','data01','data02','data03','data04','data05','data06']
datad = ['udacity','center_ccw','side2center_ccw','center_cw','side2center_cw','bridge','center_ccw2']
datan = [1,0,1,0,1,1,1]

# images filename lists
imagesCfn = []
imagesLfn = []
imagesRfn = []

# steering data lists
steeringC = []
steeringL = []
steeringR = []

# loop over each image group folder
for i in range(len(dataf)):
    image_names = np.genfromtxt('../'+dataf[i]+'/driving_log.csv', dtype=str, delimiter=',', autostrip=True, skip_header=1, usecols=(0,1,2) )
    meas        = np.genfromtxt('../'+dataf[i]+'/driving_log.csv', delimiter=',', skip_header=1, usecols=(3,4,5,6) )
    # loop to augment a folder by N times if needed (also turns off some folder data if 0)
    for j in range(datan[i]):
        # find all images in folder and append them to lists
        for k in range(image_names.shape[0]):
            # center camera
            filename = '../'+dataf[i]+'/'+image_names[k,0]
            imagesCfn.append(filename)
            steeringC.append(meas[k,0])
            # left camera with correction
            filename = '../'+dataf[i]+'/'+image_names[k,1]
            imagesLfn.append(filename)
            steeringL.append(meas[k,0] + augment_cameras_LR_factor)            
            # right camera with correction
            filename = '../'+dataf[i]+'/'+image_names[k,2]
            imagesRfn.append(filename)
            steeringR.append(meas[k,0] - augment_cameras_LR_factor)   

# augment
if augment_cameras_LR:
    imagesfn = imagesCfn + imagesLfn + imagesRfn
    steering = steeringC + steeringL + steeringR
else:
    imagesfn = imagesCfn
    steering = steeringC

# plot all steering data
plt.plot(steering)
plt.xlabel('Data'); plt.ylabel('Steering'); plt.title('Steering Input')
plt.savefig(fname+'.steering.png')
plt.show()

# histogram of steering data
str_hist, str_hist_bins = np.histogram(steering, bins = np.linspace(-1.05,1.05,20))
plt.hist(steering,bins=np.linspace(-1.05,1.05,20), edgecolor='k', alpha = 0.5, color= 'b')
plt.xlabel('Steering Input'); plt.ylabel('Count'); plt.title('Steering Training Data Histogram')
plt.grid(True)
plt.savefig(fname+'.histogram.png')
plt.show()

# put all images + steer into a tupple
samples =list(zip(imagesfn,steering))

# split 80/20
train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size=0.2)

# generator function to feed tensorflow data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread(name)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#if augment_flip_left2right:
#    images_flipped = np.fliplr(images)
#    steering_flipped =  np.array(steering, dtype=int)*-1.0
#    # create X and y
#    X_train = np.array(np.concatenate((images, images_flipped), axis=0))
#    y_train = np.array(np.concatenate((steering, steering_flipped), axis=0))
        

if Train:
    # intial time
    t_start=time.time()
    
    # NN model
    model = keras.models.Sequential()
   
    # input images are 320x160, crop 70 from top AND 25 from bottom => image is 320x65
    model.add(keras.layers.Cropping2D(cropping=((70,25),(0,0)) , input_shape=(160,320,3)))

    # normalize colors
    model.add(keras.layers.core.Lambda(lambda x: x/255.0 - 0.5))
    
    # conv2d layers to extract features (ex. 24,5,5 is a 5x5 windows that will increase 3rd dimension to 24)
    model.add(keras.layers.convolutional.Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(keras.layers.convolutional.Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(keras.layers.convolutional.Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(keras.layers.convolutional.Convolution2D(64,3,3,                  activation="relu"))
    model.add(keras.layers.convolutional.Convolution2D(64,3,3,                  activation="relu"))
    
    # flatten from 3D to 1D
    model.add(keras.layers.Flatten())
    
    # dropout (if enabled)
    if dropout: model.add(keras.layers.Dropout(0.5))
    
    # reduce the 1D layer until single node
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Dense(50))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Dense(1))

    # solver: use Mean Square Error + Adam Optimizer
    model.compile(loss='mse', optimizer='adam')

    # record solution data    
    history = model.fit_generator(train_generator,
                                  samples_per_epoch= len(train_samples),
                                  validation_data=validation_generator,
                                  nb_val_samples=len(validation_samples),
                                  nb_epoch=7)

    # output solution data to file
    orig_stdout = sys.stdout
    f = open(fname+'.summary.txt','w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

    # plot Training Loss and Validation Loss
    open(fname+'.loss.txt','w').write(str(history.history))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(fname+'.loss.png')
    plt.show()

    # save TF model to HDF5 to be used on driving simulator
    model.save(fname+'.h5')
    
    # end time, print solution time
    t_end = time.time()  
    print("Training Time = ", t_end - t_start)