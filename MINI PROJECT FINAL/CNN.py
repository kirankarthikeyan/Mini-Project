from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
 
img_width, img_height = 224, 224

train_data_dir = 'C:/Users/Dell/OneDrive/Desktop/liver_data 1/train/'
validation_data_dir = 'C:/Users/Dell/OneDrive/Desktop/liver_data 1/test/'

nb_train_samples =32
nb_validation_samples = 6
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])



train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size)


#model.save_weights('C:/Users/Dell/OneDrive/Desktop/model_saved.h5')


from keras.preprocessing.image import load_img
import numpy as np
image = load_img('C:/Users/Dell/OneDrive/Desktop/liver_data/test/no/3.jpg', target_size=(227, 227))

#import matplotlib.pyplot as plt
#plt.imshow(image)


img = np.array(image)
img = img / 255.0
img = img.reshape(1,227,227,3)
label = model.predict_classes(img)
print("Predicted Class (0 - no , 1- yes): ", label[0][0])
