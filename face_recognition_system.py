import os
import sys
import cv2
import functionsNN
from keras.models import load_model
import face_detection
import collections

name = input("What is your name?")
# validate that the user has ran the model training process
files = os.listdir()
if 'siamese_nn.h5' not in files:
    print("Error: Pre-trained Neural Network not found!")
    print("Please run siamese_nn.py first")
    sys.exit()    

# validate that the user has ran the onboarding process
if 'true_img.png' not in files:
    print("Error: True image not found!")
    print("Please run onbarding.py first")
    sys.exit()    

# load pre-trained Siamese neural network
model = load_model('siameseModel.h5', custom_objects={'contrastive_loss': functionsNN.contrastiveLoss, 'euclidean_distance': functionsNN.euclideanDistance})

# prepare the true image obtained during onboard
trueImg = cv2.imread('true_img.png', 0)
trueImg = trueImg.astype('float32') / 255
trueImg = cv2.resize(trueImg, (92, 112))
trueImg = trueImg.reshape(1, trueImg.shape[0], trueImg.shape[1], 1)

videoCapture = cv2.VideoCapture(0)
preds = collections.deque(maxlen=15)

while True:
    # Capture frame-by-frame
    _, frame = videoCapture.read()

    # Detect Faces
    frame, face_img, faceCoords = face_detection.detect_faces(frame, drawBox=False)

    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype('float32')/255
        face_img = cv2.resize(face_img, (92, 112))
        face_img = face_img.reshape(1, face_img.shape[0], face_img.shape[1], 1)
        preds.append(1 - model.predict([trueImg, face_img])[0][0])
        x,y,w,h = faceCoords
        if len(preds) == 15 and sum(preds)/15 >= 0.3:
            text = "Identity: {}".format(name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
        elif len(preds) < 15:
            text = "Identifying ..."
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 5)
        else:
            text = "Identity Unknown!"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
        frame = functionsNN.write_on_frame(frame, text, faceCoords[0], faceCoords[1] - 10)

    else:
        preds = collections.deque(maxlen=15) # clear existing predictions if no face detected 

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
videoCapture.release()
cv2.destroyAllWindows()
