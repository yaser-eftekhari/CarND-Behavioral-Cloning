# import statements
import csv
import cv2
import numpy as np
import tensorflow as tf

# defining the flags for interactive parameter setting
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_string('path', 'data/', "Path to the sample trainings.")
flags.DEFINE_float('correction', 0.2, "Correction factor for side images.")

# Read the CSV file and parse it
lines = []
with open(FLAGS.path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Remove the header of the csv file (legends)
del lines[0]

# Read the center, left and right images along with the corresponding measurements
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = FLAGS.path + 'IMG/' + filename

        image_rgb = cv2.imread(current_path)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YUV)
        images.append(image)

    measurement = float(line[3])
    # Use the read measurement for the center image
    measurements.append(measurement)
    # Adjust the read measurement for the left image
    measurements.append(measurement + FLAGS.correction)
    # Adjust the read measurement for the right image
    measurements.append(measurement - FLAGS.correction)

# Augment training images by flipping all images so far horizontally
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)

    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

# Create np arrays from the measurements and images
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Create the Keras model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Normalization layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = X_train.shape[1:]))
# Cropping layer
model.add(Cropping2D(cropping=((70, 25), (0,0))))
# 3 Convolutional layers of size 5x5
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
# 2 Convolutional layers of size 3x3
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
# Flattening layer
model.add(Flatten())
# 3 Fully connected layers followed by the output layer
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Optimizing for mse metric using Adam optimizer
model.compile(optimizer='adam', loss='mse')

# Fitting the model with training and validation data (a split of 20%)
model.fit(X_train, y_train, nb_epoch = FLAGS.epochs, validation_split = 0.2, shuffle = True)

# Save the model
model.save('model.h5')
exit()
