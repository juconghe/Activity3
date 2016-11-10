# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: cs390mb

Assignment 3 : Speaker Identification

This is the starter script for training a model for identifying
speaker from audio data. The script loads all labelled speaker
audio data files in the specified directory. It extracts features
from the raw data and trains and evaluates a classifier to identify
the speaker.

"""

import os
import sys
import numpy as np

# The following are classifiers you may be interested in using:
from sklearn.tree import DecisionTreeClassifier # decision tree classifier
from sklearn.ensemble import RandomForestClassifier # random forest classifier
from sklearn.neighbors import NearestNeighbors # k-nearest neighbors (k-NN) classiifer
from sklearn.svm import SVC #SVM classifier

from features import FeatureExtractor
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pickle


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

data_dir = 'data' # directory where the data files are stored

output_dir = 'training_output' # directory where the classifier(s) are stored

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# the filenames should be in the form 'speaker-data-subject-1.csv', e.g. 'speaker-data-Erik-1.csv'. If they
# are not, that's OK but the progress output will look nonsensical

class_names = [] # the set of classes, i.e. speakers

data = np.zeros((0,8002)) #8002 = 1 (timestamp) + 8000 (for 8kHz audio data) + 1 (label)

for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("speaker-data"):
        filename_components = filename.split("-") # split by the '-' character
        speaker = filename_components[2]
        print("Loading data for {}.".format(speaker))
        if speaker not in class_names:
            class_names.append(speaker)
        speaker_label = class_names.index(speaker)
        sys.stdout.flush()
        data_file = os.path.join('data', filename)
        data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
        print("Loaded {} raw labelled audio data samples.".format(len(data_for_current_speaker)))
        sys.stdout.flush()
        data = np.append(data, data_for_current_speaker, axis=0)

print("Found data for {} speakers : {}".format(len(class_names), ", ".join(class_names)))

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# You may need to change n_features depending on how you compute your features
# we have it set to 3 to match the dummy values we return in the starter code.
n_features = 995

print("Extracting features and labels for {} audio windows...".format(data.shape[0]))
sys.stdout.flush()

X = np.zeros((0,n_features))
y = np.zeros(0,)

# change debug to True to show print statements we've included:
feature_extractor = FeatureExtractor(debug=False)

for i,window_with_timestamp_and_label in enumerate(data):
    window = window_with_timestamp_and_label[1:-1] # get window without timestamp/label
    label = data[i,-1] # get label
    x = feature_extractor.extract_features(window)  # extract features

    # if # of features don't match, we'll tell you!
    if (len(x) != X.shape[1]):
        print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))

    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    y = np.append(y, label)

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()


# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)
sum_conf = []

# TODO: Train your classifier!
tree = DecisionTreeClassifier (criterion = "gini" ,  max_depth = 3)
cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
for i, (train_indexes, test_indexes) in enumerate(cv):
    print("Fold {} : The confusion matrix is :".format(i))
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    if(sum_conf == []):
        sum_conf = confusion_matrix(y_test,y_pred)
    else:
        sum_conf += confusion_matrix(y_test,y_pred)
    print(sum_conf)

accuracy = sum(np.diagonal(sum_conf))/(np.sum(sum_conf)*1.0)
print("Average accuracy is {}".format(accuracy))
precision_Jucong= sum_conf[0,0]/(sum(sum_conf[:, 0])*1.0)
print("Precision of Jucong Speaking is {}".format(precision_Jucong))
recall_Jucong = sum_conf[0,0]/(sum(sum_conf[0])*1.0)
print("Recall of Jucong Speaking is {}".format(recall_Jucong))
precision_Xin = sum_conf[1,1]/(sum(sum_conf[:, 1])*1.0)
print("Precision of Xin is {}".format(precision_Xin))
recall_Xin= sum_conf[1,1]/(sum(sum_conf[1])*1.0)
print("Recall of Xin is {}".format(recall_Xin))
precision_Karen = sum_conf[2,2]/(sum(sum_conf[:, 2])*1.0)
print("Precision of Karen is {}".format(precision_Karen))
recall_Karen = sum_conf[2,2]/(sum(sum_conf[2])*1.0)
print("Recall of Karen is {}".format(recall_Karen))
# precision_NoSpeaker = sum_conf[3,3]/(sum(sum_conf[:, 3])*1.0)
# print("Precision of No Speaker is {}".format(precision_NoSpeaker))
# recall_NoSpeaker = sum_conf[3,3]/(sum(sum_conf[3])*1.0)
# print("Recall of No Speaker is {}".format(recall_NoSpeaker))

print("=======SVC classifier=========")
sum_conf = []
svc = SVC()
for i, (train_indexes, test_indexes) in enumerate(cv):
    print("Fold {} : The confusion matrix is :".format(i))
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    if(sum_conf == []):
        sum_conf = confusion_matrix(y_test,y_pred)
    else:
        sum_conf += confusion_matrix(y_test,y_pred)
    print(sum_conf)

accuracy = sum(np.diagonal(sum_conf))/(np.sum(sum_conf)*1.0)
print("Average accuracy is {}".format(accuracy))
precision_Jucong= sum_conf[0,0]/(sum(sum_conf[:, 0])*1.0)
print("Precision of Jucong Speaking is {}".format(precision_Jucong))
recall_Jucong = sum_conf[0,0]/(sum(sum_conf[0])*1.0)
print("Recall of Jucong Speaking is {}".format(recall_Jucong))
precision_Xin = sum_conf[1,1]/(sum(sum_conf[:, 1])*1.0)
print("Precision of Xin is {}".format(precision_Xin))
recall_Xin= sum_conf[1,1]/(sum(sum_conf[1])*1.0)
print("Recall of Xin is {}".format(recall_Xin))
precision_Karen = sum_conf[2,2]/(sum(sum_conf[:, 2])*1.0)
print("Precision of Karen is {}".format(precision_Karen))
recall_Karen = sum_conf[2,2]/(sum(sum_conf[2])*1.0)
print("Recall of Karen is {}".format(recall_Karen))
# precision_NoSpeaker = sum_conf[3,3]/(sum(sum_conf[:, 3])*1.0)
# print("Precision of No Speaker is {}".format(precision_NoSpeaker))
# recall_NoSpeaker = sum_conf[3,3]/(sum(sum_conf[3])*1.0)
# print("Recall of No Speaker is {}".format(recall_NoSpeaker))


print("=======rfc========")
sum_conf = []
rfc = RandomForestClassifier()
for i, (train_indexes, test_indexes) in enumerate(cv):
    print("Fold {} : The confusion matrix is :".format(i))
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    rfc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    if(sum_conf == []):
        sum_conf = confusion_matrix(y_test,y_pred)
    else:
        sum_conf += confusion_matrix(y_test,y_pred)
    print(sum_conf)

accuracy = sum(np.diagonal(sum_conf))/(np.sum(sum_conf)*1.0)
print("Average accuracy is {}".format(accuracy))
precision_Jucong= sum_conf[0,0]/(sum(sum_conf[:, 0])*1.0)
print("Precision of Jucong Speaking is {}".format(precision_Jucong))
recall_Jucong = sum_conf[0,0]/(sum(sum_conf[0])*1.0)
print("Recall of Jucong Speaking is {}".format(recall_Jucong))
precision_Xin = sum_conf[1,1]/(sum(sum_conf[:, 1])*1.0)
print("Precision of Xin is {}".format(precision_Xin))
recall_Xin= sum_conf[1,1]/(sum(sum_conf[1])*1.0)
print("Recall of Xin is {}".format(recall_Xin))
precision_Karen = sum_conf[2,2]/(sum(sum_conf[:, 2])*1.0)
print("Precision of Karen is {}".format(precision_Karen))
recall_Karen = sum_conf[2,2]/(sum(sum_conf[2])*1.0)
print("Recall of Karen is {}".format(recall_Karen))
# precision_NoSpeaker = sum_conf[3,3]/(sum(sum_conf[:, 3])*1.0)
# print("Precision of No Speaker is {}".format(precision_NoSpeaker))
# recall_NoSpeaker = sum_conf[3,3]/(sum(sum_conf[3])*1.0)
# print("Recall of No Speaker is {}".format(recall_NoSpeaker))


print("======nearest Neighbors========")
nnb = NearestNeighbors()
sum_conf = []
for i, (train_indexes, test_indexes) in enumerate(cv):
    print("Fold {} : The confusion matrix is :".format(i))
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    nnb.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    conf = confusion_matrix(y_test,y_pred)
    if(sum_conf == []):
        sum_conf = confusion_matrix(y_test,y_pred)
    else:
        sum_conf += confusion_matrix(y_test,y_pred)
    print(sum_conf)

accuracy = sum(np.diagonal(sum_conf))/(np.sum(sum_conf)*1.0)
print("Average accuracy is {}".format(accuracy))
precision_Jucong= sum_conf[0,0]/(sum(sum_conf[:, 0])*1.0)
print("Precision of Jucong Speaking is {}".format(precision_Jucong))
recall_Jucong = sum_conf[0,0]/(sum(sum_conf[0])*1.0)
print("Recall of Jucong Speaking is {}".format(recall_Jucong))
precision_Xin = sum_conf[1,1]/(sum(sum_conf[:, 1])*1.0)
print("Precision of Xin is {}".format(precision_Xin))
recall_Xin= sum_conf[1,1]/(sum(sum_conf[1])*1.0)
print("Recall of Xin is {}".format(recall_Xin))
precision_Karen = sum_conf[2,2]/(sum(sum_conf[:, 2])*1.0)
print("Precision of Karen is {}".format(precision_Karen))
recall_Karen = sum_conf[2,2]/(sum(sum_conf[2])*1.0)
print("Recall of Karen is {}".format(recall_Karen))
# precision_NoSpeaker = sum_conf[3,3]/(sum(sum_conf[:, 3])*1.0)
# print("Precision of No Speaker is {}".format(precision_NoSpeaker))
# recall_NoSpeaker = sum_conf[3,3]/(sum(sum_conf[3])*1.0)
# print("Recall of No Speaker is {}".format(recall_NoSpeaker))

best_classifier = rfc
classifier_filename='classifier.pickle'
print("Saving best classifier to {}...".format(os.path.join(output_dir, classifier_filename)))
with open(os.path.join(output_dir, classifier_filename), 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)
