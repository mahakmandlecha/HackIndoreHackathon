import pandas as pd
import numpy as np
import glob
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import os,shutil
from keras import models,layers,optimizers
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


df= pd.read_csv('train.csv')
#df.replace(0,'cardboard').replace(1,'plastic').replace(2,'wet_waste')
datagen = ImageDataGenerator(rescale=1./255,
                           shear_range = 0.2,
                           zoom_range = 0.2,
                           horizontal_flip=True,
                           preprocessing_function= preprocess_input,
                           validation_split=0.25)
train_generator = datagen.flow_from_dataframe(dataframe=df,
                        directory="C:/Users/admin/Desktop/mlfinal/training",
                        x_col="image_name",
                        y_col= 'x',
                        subset="training",
                        batch_size=100,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(150,150))
valid_generator = datagen.flow_from_dataframe(dataframe=df,
                        directory="C:/Users/admin/Desktop/mlfinal/training",
                        x_col="image_name",
                        y_col= 'x',
                        subset="validation", batch_size=10,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(150,150))

from keras.applications.resnet50 import ResNet50
res_conv = ResNet50(include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=(150,150,3),
                    pooling=None,classes=1000)
model = models.Sequential()
model.add(res_conv)

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Show a summary of the model. Check the number of trainable parameters
history = model.fit_generator(
    train_generator, 
    epochs=20,
    validation_data = valid_generator,
    #validation_steps= len(valid_generator),
    steps_per_epoch = len(train_generator)
)




import matplotlib.pyplot as plt
#Loss
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['train', 'test'], loc='upper left') 
plt.show()
#Accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#DATATESTING
"""
def img_show(image):
  b,g,r = cv2.split(image)
  image = cv2.merge((r,g,b))
  plt.imshow(image)
  plt.show()
  return image
def test(model,image_path):
  img = cv2.imread(image_path)
  img = img_show(img)
  img = cv2.resize(img,(224,224))
  img = np.reshape(img,(1,224,224,3))
  img = img/255.0
  prediction = model.predict(img)
  prediction = np.argmax(prediction)
  labels = (train_generator.class_indices)
  labels = dict((v,k) for k,v in labels.items())
  return labels[prediction]

print(train_generator.class_indices)
#printed output
{'CARDBOARD': 0, 'PLASTIC': 1, 'WETWASTE': 2}
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)
#Printed output
{0:'CARDBOARD', 1:'PLASTIC', 2:'WETWASTE'}
"""