# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:09:49 2024

@author: This Pc
"""
import os

os.chdir(r"E:\Project_2024\All Python Projects\Deep Learning Project\PlantVillage")
os.listdir()

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 25

dataset = tf.keras.preprocessing.image_dataset_from_directory(r"E:\Project_2024\All Python Projects/Deep Learning Project\PlantVillage",
                                                    shuffle=True,
                                                    image_size = (IMAGE_SIZE,IMAGE_SIZE),
                                                    batch_size = BATCH_SIZE
                                                    )
#Output Found 2152 files belonging to 3 classes.

class_names = dataset.class_names
class_names

#Output ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

len(dataset)

# 68

for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())
    
#Output 
#(32, 256, 256, 3)
#[1 1 2 2 0 0 0 2 1 1 1 1 1 1 1 1 1 0 0 1 0 1 2 0 0 0 0 1 1 1 1 1] 

#printing first image
plt.figure(figsize=(10,10))   
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.axis("off")
        plt.title(class_names[label_batch[i]])
 

#(256,256,3)

#train - test split 

#80% == >training
#20% == >10% validation, 10% test

train_size= 0.8
len(dataset)*train_size   #54.40000

train_ds = dataset.take(54)
len(train_ds)

#54

test_ds = dataset.skip(54)
len(test_ds)

#14

val_size = 0.1
len(dataset)*val_size  #6.800
val_ds = test_ds.take(6)
len(val_ds)

#6

test_ds = test_ds.skip(6)
len(test_ds)

#8

def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size,seed=12)
    
    train_size = int(train_split * ds_size)
    val_size= int(val_split * ds_size)
    
    train_ds= ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)    
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
len(train_ds)  #54
len(val_ds)    #8
len(test_ds)   #6

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)  
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)   
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE) 

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
]) 


train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True),y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3,3),activation='relu',input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size= (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
    ])

model.build(input_shape=input_shape)

model.summary()

#Total params: 183,747
#Trainable params: 183,747
#Non-trainable params: 0

model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size= BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

scores = model.evaluate(test_ds)
#loss: 0.0371 - accuracy: 0.9890 - val_loss: 0.1795 - val_accuracy: 0.9531

scores  #Out[48]: [0.28674110770225525, 0.8984375]
# Out[31]: [0.06012701243162155, 0.98046875]

history
history.params  #Out[51]: {'verbose': 1, 'epochs': 10, 'steps': 54}
history.history.keys()  #Out[52]: dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

acc = history.history['accuracy']
val_acc=history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),acc, label='Training Accuracy')
plt.plot(range(EPOCHS),val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

#Out[57]: Text(0.5, 1.0, 'Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(EPOCHS),loss, label='Training loss')
plt.plot(range(EPOCHS),val_loss,label='Validation loss')
plt.legend(loc='upper right')
plt.title('Training and Validation loss')
plt.show()

for images_batch,labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0]
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction =model.predict(images_batch)
    print(np.argmax(batch_prediction[0]))
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
    
np.argmax([])  #values from above code

def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0) # create a batch
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence =round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence

plt.figure(figsize=(15,15))
for images,labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model,images[i].numpy())
        actual_class = class_names[labels[i]]
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class},\n Confidence: {confidence}% ")
        
        plt.axis("off")
        

# Define the base directory to save the models
base_dir = r"E:\Project_2024\All Python Projects\Deep Learning Project\Models01"

# Ensure the base directory exists, if not, create it
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Define the number of versions to save
num_versions = 3

# Function to save multiple versions
def save_multiple_versions(model, base_dir, num_versions):
    # Save model for each version from 1 to num_versions
    for version in range(1, num_versions + 1):
        version_dir = os.path.join(base_dir, str(version))
        
        # Ensure the directory for this version exists, if not, create it
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
        
        # Save the model in the version directory
        model.save(os.path.join(version_dir, 'Models1'))
        print(f"Model saved as version {version} in {version_dir}")

# Example usage
# model = ...  # your model creation or loading code here
# save_multiple_versions(model, base_dir, num_versions)


