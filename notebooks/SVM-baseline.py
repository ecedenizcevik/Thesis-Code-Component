#!/usr/bin/env python
# coding: utf-8

# In[3]:


# SVM Baseline Notebook for Facial Emotion Recognition

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import joblib

# I split the dataset into train and test with a separate script which can be found in the Git repository.
dirs = {'train': '../dataset_split/train', 'test': '../dataset_split/test'}
batch_size = 32
img_size   = (224,224)

def print_header(text):
    print("\n" + "="*len(text))
    print(text)
    print("="*len(text))

# I load ResNet50 without its top layers so that I can use its pretrained filters as feature extractors.
print_header("Feature Extraction with ResNet50")
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#I resize the images for model and flatten to 1D.
def extract_features(directory):
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    print(f"Found classes: {classes}")
    features, labels = [], []
    class_idx = {cls: i for i, cls in enumerate(classes)}
    for cls in classes:
        cls_dir = os.path.join(directory, cls)
        for fname in tqdm(os.listdir(cls_dir), desc=f"Processing {cls}"):
            path = os.path.join(cls_dir, fname)
            try:
                img = image.load_img(path, target_size=img_size)
                arr = image.img_to_array(img)
                arr = preprocess_input(arr)
                feats = base_model.predict(np.expand_dims(arr, axis=0), verbose=0)
                features.append(feats.flatten())
                labels.append(class_idx[cls])
            except Exception:
                continue
    X = np.array(features)
    y = np.array(labels)
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features")
    return X, y, classes

X_train, y_train, class_names = extract_features(dirs['train'])
X_test,  y_test,  _           = extract_features(dirs['test'])

# I standardize to zero mean/unit variance so PCA and SVM behave better.
print_header("Preprocessing: Scaling + PCA")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# I reduce to 200 PCA components (chosen to preserve >90% variance) to speed up SVM training and reduce noise.
pca = PCA(n_components=200, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)
print(f"Reduced to {X_train_pca.shape[1]} PCA components")

# I use an RBF-kernel SVM with balanced class weights and probability estimates, and train the model.
print_header("Training SVM Model")
svm = SVC(
    kernel='rbf', C=1, gamma='scale',
    class_weight='balanced', probability=True, verbose=True
)
svm.fit(X_train_pca, y_train)
print("SVM training complete.")

# I create plots for visualization.
print_header("Learning Curve")
train_sizes, train_scores, val_scores = learning_curve(
    svm, X_train_pca, y_train,
    train_sizes=np.linspace(0.1,1.0,5),
    cv=5, scoring='accuracy', n_jobs=-1, verbose=0
)
train_mean = np.mean(train_scores, axis=1)
val_mean   = np.mean(val_scores,   axis=1)
plt.figure(figsize=(6,4))
plt.plot(train_sizes * len(X_train_pca), train_mean, label='Train Acc')
plt.plot(train_sizes * len(X_train_pca), val_mean,   label='Val Acc')
plt.title('SVM(baseline) Learning Curve')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

proba = svm.predict_proba(X_test_pca)
y_test_binarized = label_binarize(y_test, classes=range(len(class_names)))
plt.figure(figsize=(8, 6))
for i, cls in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(
        y_test_binarized[:, i],
        proba[:, i]
    )
    ap_score = average_precision_score(
        y_test_binarized[:, i],
        proba[:, i]
    )
    plt.plot(recall, precision, lw=2,
             label=f"{cls} (AP = {ap_score:.2f})")
plt.title("SVM(baseline) Precision–Recall Curves by Class")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final evaluation on the test set (accuracy and log-loss are printed here for completeness).
print_header("Evaluation on Test Set")
proba = svm.predict_proba(X_test_pca)
preds = svm.predict(X_test_pca)
acc   = svm.score(X_test_pca, y_test)
ll    = log_loss(y_test, proba)
print(f"Test Accuracy : {acc:.4f}")
print(f"Test Log Loss : {ll:.4f}")
print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('SVM(baseline) Confusion Matrix')
plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.tight_layout(); plt.show()

# I save the final model
print_header("Saving Model and Preprocessor")
joblib.dump({'svm': svm, 'scaler': scaler, 'pca': pca}, '../models/svm_pipeline_simple.joblib')
print("Artifacts saved to ../models/svm_pipeline_simple.joblib")


# In[ ]:




