#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Custom-CNN Notebook for Facial Emotion Recognition

import os, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    BatchNormalization, Dropout,
    Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


# I use ImageDataGenerator to scale pixels to [0,1] and randomly rotate, shift, zoom, etc., so that the model learns robust features.
# I also reserve 20% of train data for the validation split.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

# I split the dataset into train and test with a separate script which can be found in the Git repository.
train_dir = '../dataset_split/train'
test_dir  = '../dataset_split/test'
img_size  = (128,128)
batch_size = 32

# I load images in batches.
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='training', shuffle=True
)
val_data = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='validation', shuffle=True
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', shuffle=False
)

# I use four convolutional layers, each followed by BatchNormalization and dropout to extract rich features while preventing overfitting.
# And I use a small L2 regularization to improve generalization.
reg = regularizers.l2(1e-5)  
model = Sequential([
    Input(shape=(*img_size, 3)),

    Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=reg),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.20),            

    Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=reg),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.20),

    Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=reg),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.20),

    Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=reg),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.20),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=reg), 
    Dropout(0.40),                                        
    Dense(6, activation='softmax')
])

# I compile the model with Adam optimizer for smooth learning and track accuracy.
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# I use EarlyStopping to avoid overfitting, ModelCheckpoint to keep the best weights, and ReduceLROnPlateau to lower the learning rate when progress stalls.
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint('../models/best_cnn_model.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

# I train the model, aiming for up to 60 epochs, and I also record elapsed training time.
start = time.time()
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=60,               
    callbacks=callbacks
)
print(f"Elapsed: {(time.time()-start)/60:.1f} min")

# I create plots for visualization.
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'],   label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Custom-CNN Training & Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'],       label='Train Loss')
plt.plot(history.history['val_loss'],   label='Val Loss')
plt.title('Custom-CNN Training & Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True); plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

test_data.reset()
preds = model.predict(test_data)
y_true = test_data.classes
labels = list(test_data.class_indices.keys())
y_true_binarized = label_binarize(y_true, classes=range(len(labels)))
plt.figure(figsize=(8, 6))
for i, cls in enumerate(labels):
    precision, recall, _ = precision_recall_curve(
        y_true_binarized[:, i],
        preds[:, i]
    )
    ap = average_precision_score(
        y_true_binarized[:, i],
        preds[:, i]
    )
    plt.plot(recall, precision, lw=2,
             label=f"{cls} (AP={ap:.2f})")

plt.title("Custom-CNN Precision–Recall Curves by Class")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final evaluation on test set.
test_loss, test_acc = model.evaluate(test_data, verbose=1)
print(f"Test Accuracy: {test_acc*100:.2f}%  —  Loss: {test_loss:.3f}")

test_data.reset()
preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes
labels = list(test_data.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Custom-CNN Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.show()

#I save the final model
model.save('../models/custom_cnn_emotion.h5')


# In[ ]:




