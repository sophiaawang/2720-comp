#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVC
import pandas as pd

from scipy.io import wavfile
from scipy.fft import rfft, irfft

import csv

# Getting the training set's labels
label_path = "/classes/ece2720/fpc/train.csv"
train_csv = pd.read_csv(label_path)
labels = train_csv["Label"].to_numpy()

train_set_path = "/classes/ece2720/fpc/train/train_"
test_set_path = "/classes/ece2720/fpc/test/test_"

# Getting training data:
all_frames = np.zeros((48000, 4000))

for i in range(48000):
    train_path = train_set_path + str(i) + ".wav"
    rate, frames = wavfile.read(train_path)
    all_frames[i] = frames

# Getting test data:
all_test_frames = np.zeros((16000, 4000))

for i in range(16000):
    test_path = test_set_path + str(i) + ".wav"
    rate, frames = wavfile.read(test_path)
    all_test_frames[i] = frames


# Fourier Transform -

# Transformed features:
all_frames_transformed = []
all_test_frames_transformed = []

# Dft for training set
print("transforming train")
all_frames_rfft = rfft(all_frames)
for i in range(len(all_frames)):
    # Setting to 0 anything with a frequency over 535 or under 50
    all_frames_rfft[i][535:] = 0
    all_frames_rfft[i][0:50] = 0

    all_frames_irfft = irfft(all_frames_rfft[i])
    all_frames_transformed.append(all_frames_irfft)

# Dft for test set
print("transforming test")
all_test_frames_rfft = rfft(all_test_frames)
for i in range(len(all_test_frames)):
    all_test_frames_rfft[i][535:] = 0
    all_test_frames_rfft[i][0:50] = 0

    all_test_frames_irfft = irfft(all_test_frames_rfft[i])
    all_test_frames_transformed.append(all_test_frames_irfft)

all_frames_transformed = np.array(all_frames_transformed)
all_test_frames_transformed = np.array(all_test_frames_transformed)


# Running an SVC with C=10 on all training data
print("training")
svm_model_fft = svm.SVC(C=10, break_ties=True)

print("fitting")
svm_model_fft.fit(all_frames_transformed, labels)

print("predicting")
test_pred_fft = svm_model_fft.predict(all_test_frames_transformed)


# Creating the csv for submission
fft_indices = np.arange(16000)
fft_predictions = list(zip(fft_indices, test_pred_fft))
fft_predictions_sorted = sorted(fft_predictions, key=lambda x: x[0])
fft_file_name = "tuned_fft_50-535_c10.csv"

with open(fft_file_name, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["# ID", "Label"])
    writer.writerows(fft_predictions_sorted)
