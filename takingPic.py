import functionsNN
import detectingFace
import cv2
import math


recordVideo = cv2.VideoCapture(0)
timer1 = 5
while True:
    text = 'Cheese! {}..'.format(math.ceil(timer1))
    _, frame = recordVideo.read()
    frame, face_box, faceCoords = detectingFace.detectFaces(frame)
    if face_box is not None:
        frame = functionsNN.write_on_frame(frame, text, faceCoords[0], faceCoords[1] - 10)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    timer1 -= 0.1
    if timer1 <= 0:
        cv2.imwrite('true_img.png', face_box)
        break
recordVideo.release()
cv2.destroyAllWindows()

