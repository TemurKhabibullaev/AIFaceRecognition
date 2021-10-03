import os
import sys
import detectingFace
import collections
import cv2
from keras.models import load_model
import functionsNN

usersName = input("Input your name:\n")
files = os.listdir()
if 'true_img.png' not in files:
    print("Your face picture was not found! Please take a picture first!")
    sys.exit()    
if 'siameseModel.h5' not in files:
    print("You haven't trained Seamese NN model!")
    sys.exit()
model = load_model('siameseModel.h5', custom_objects={'contrastiveLoss': functionsNN.contrastiveLoss, 'euclideanDistance': functionsNN.euclideanDistance})
trueImg = cv2.imread('true_img.png', 0)
trueImg = trueImg.astype('float32') / 255
trueImg = cv2.resize(trueImg, (92, 112))
trueImg = trueImg.reshape(1, trueImg.shape[0], trueImg.shape[1], 1)
videoCapture = cv2.VideoCapture(0)
preds = collections.deque(maxlen=15)
while True:
    # Here we are recording frame by frame and detecting a face
    _, frame = videoCapture.read()
    frame, face_img, faceCoords = detectingFace.detectFaces(frame, drawBox=False)

    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype('float32')/255
        face_img = cv2.resize(face_img, (92, 112))
        face_img = face_img.reshape(1, face_img.shape[0], face_img.shape[1], 1)
        preds.append(1 - model.predict([trueImg, face_img])[0][0])
        x, y, w, h = faceCoords
        if len(preds) == 15 and sum(preds)/15 >= 0.3:
            text = "Name: {}".format(usersName)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
        elif len(preds) < 15:
            text = "Hmmm ..."
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 5)
        else:
            text = "Unknown Identity!"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
        frame = functionsNN.write_on_frame(frame, text, faceCoords[0], faceCoords[1] - 10)
    else:
        preds = collections.deque(maxlen=15)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
