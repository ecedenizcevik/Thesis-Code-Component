#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# I mount Google Drive to access the dataset archive in Colab Environment.
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#I zip my split dataset in Drive(script used for splitting the dataset can be found in the Git repository).
get_ipython().system('cd /tmp && zip -r -X dataset_split_clean.zip dataset_split')
get_ipython().system('cp /tmp/dataset_split_clean.zip "/content/drive/MyDrive/thesis_code/code/"')


# In[ ]:


# VGG16 Transfer-Learning Notebook for Facial Emotion Recognition

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix

# I unzip cleaned dataset into /tmp and remove macOS __MACOSX folders, resulting in the same train/validation/test split used elsewhere.
get_ipython().system('rm -rf /tmp/dataset_split')
get_ipython().system('unzip -q "/content/drive/MyDrive/thesis_code/code/dataset_split_clean.zip" -d /tmp')
get_ipython().system('rm -rf /tmp/dataset_split/__MACOSX')
if os.path.isdir('/tmp/dataset_split/dataset_split'):
    for entry in os.listdir('/tmp/dataset_split/dataset_split'):
        src = os.path.join('/tmp/dataset_split/dataset_split', entry)
        dst = os.path.join('/tmp/dataset_split', entry)
        os.replace(src, dst)
    get_ipython().system('rm -rf /tmp/dataset_split/dataset_split')

data_dir   = "/tmp/dataset_split"
train_dir  = os.path.join(data_dir, 'train')
test_dir   = os.path.join(data_dir, 'test')

# I use ImageDataGenerator to scale pixels to [0,1] and randomly rotate, shift, zoom, etc., so that the model learns robust features.
# I also reserve 20% of train data for the validation split(hold out 20% of training images for validation).
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=(0.8,1.2),
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# I use ImageDataGenerator for test data but only preprocessing and no augmentation.
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# I load images in batches.
target_size = (224, 224)
batch_size  = 32

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=target_size,
    batch_size=batch_size, class_mode='categorical',
    subset='training', shuffle=True
)
val_data = train_datagen.flow_from_directory(
    train_dir, target_size=target_size,
    batch_size=batch_size, class_mode='categorical',
    subset='validation', shuffle=True
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=target_size,
    batch_size=batch_size, class_mode='categorical',
    shuffle=False
)

# I initialize VGG16 backbone and freeze all but the last two convolutional layers so that only the high-level, task-specific filters are fine-tuned.
base = VGG16(weights='imagenet', include_top=False,
             input_shape=(224, 224, 3))
for layer in base.layers[:-2]:
    layer.trainable = False

# I add a custom classification head with regularization.
x = GlobalAveragePooling2D()(base.output)
x = BatchNormalization()(x)
x = Dense(
    512, activation='relu',
    kernel_regularizer=regularizers.l2(1e-4)
)(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
outputs = Dense(
    train_data.num_classes, activation='softmax'
)(x)

# I compile the model with low learning rate for fine-tuning.
model = Model(inputs=base.input, outputs=outputs)
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# I use EarlyStopping to avoid overfitting, ModelCheckpoint to keep the best weights, and ReduceLROnPlateau to lower the learning rate when progress stalls.
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_vgg16_finetuned.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# I train the model and aim for up to 50 epochs but EarlyStopping ends on the 24th epoch.
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=callbacks
)

# I create plots fro completeness .
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'],   label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('VGG16 Training vs. Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'],   label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('VGG16 Training vs. Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# Final evaluation on test set.
test_data.reset()
preds = model.predict(test_data)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_data.classes
labels = list(test_data.class_indices.keys())

print(classification_report(true_classes, pred_classes, target_names=labels))
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('VGG16 Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.show()

#I save the final model.
model.save('vgg16_finetuned_emotion.h5')


# In[ ]:


from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# I plot per-class precision–recall curves with average precision scores.
y_true_binarized = label_binarize(true_classes, classes=range(len(labels)))
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

plt.title("Precision–Recall Curves by Class (VGG16)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()

