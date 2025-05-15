#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# I use this script to split my dataset into train/test subfolders for easy use with ImageDataGenerator.

import os
import shutil
import random

def split_dataset(original_data_dir, base_output_dir, train_split=0.8):
    train_dir = os.path.join(base_output_dir, 'train')
    test_dir = os.path.join(base_output_dir, 'test')

    
    for folder in [train_dir, test_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

# I gather all images, shuffle for randomness, and split.
    for class_name in os.listdir(original_data_dir):
        class_path = os.path.join(original_data_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)

            split_index = int(len(images) * train_split)

            train_images = images[:split_index]
            test_images = images[split_index:]

            
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # I copy images to train
            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(train_dir, class_name, img)
                shutil.copy2(src, dst)

            # And the rest to test
            for img in test_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(test_dir, class_name, img)
                shutil.copy2(src, dst)

    print("Done")


original_dataset_dir = 'dataset'
output_base_dir = 'dataset_split'

#80% train, 20% test by default
split_dataset(original_dataset_dir, output_base_dir)



# In[ ]:




