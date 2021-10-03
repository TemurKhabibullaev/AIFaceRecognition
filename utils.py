# All necessary functions for NN
import os
import cv2
import numpy as np
import random
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Sequential


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def euclideanDistance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def createdPairs(x, y, numClasses):
    pairs, labels = [], []
    class_idx = [np.where(y == i)[0] for i in range(numClasses)]
    min_images = min(len(class_idx[i]) for i in range(numClasses)) - 1
  
    for c in range(numClasses):
        for n in range(min_images):
            img1 = x[class_idx[c][n]]
            img2 = x[class_idx[c][n + 1]]
            pairs.append((img1, img2))
            labels.append(1)
            negList = list(range(numClasses))
            negList.remove(c)
            negC = random.sample(negList, 1)[0]
            img1 = x[class_idx[c][n]]
            img2 = x[class_idx[negC][n]]
            pairs.append((img1, img2))
            labels.append(0)
    return np.array(pairs), np.array(labels)


def contrastiveLoss(YTrue, D):
    margin = 1
    return K.mean(YTrue * K.square(D) + (1 - YTrue) * K.maximum((margin - D), 0))


def sharedNetwork(inputShape):
    model = Sequential(name='Shared_Conv_Network')
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid'))
    return model


def write_on_frame(frame, text, textX, textY):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
    boxCoords = ((textX, textY), (textX + text_width + 20, textY - text_height - 20))
    cv2.rectangle(frame, boxCoords[0], boxCoords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, text, (textX, textY - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame


def get_data(dir):
    XTrain, YTrain = [], []
    XTest, YTest = [], []
    subfolders = sorted([file.path for file in os.scandir(dir) if file.is_dir()])
    for idx, folder in enumerate(subfolders):
        for file in sorted(os.listdir(folder)):
            img = load_img(folder+"/"+file, color_mode='grayscale')
            img = img_to_array(img).astype('float32')/255
            img = img.reshape(img.shape[0], img.shape[1], 1)
            if idx < 35:
                XTrain.append(img)
                YTrain.append(idx)
            else:
                XTest.append(img)
                YTest.append(idx-35)

    XTrain = np.array(XTrain)
    XTest = np.array(XTest)
    YTrain = np.array(YTrain)
    YTest = np.array(YTest)
    return (XTrain, YTrain), (XTest, YTest)





