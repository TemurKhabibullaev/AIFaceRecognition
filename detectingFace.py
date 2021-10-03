import os
import cv2
face_cascade = cv2.CascadeClassifier('frontFace.xml')


def detectFaces(img, drawBox=True):
	grayscaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(grayscaleImg, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	faceBox, faceCoords = None, []
	for (a, b, c, d) in faces:
		if drawBox:
			cv2.rectangle(img, (a, b), (a+c, b+d), (0, 255, 0), 5)
		faceBox = img[b:b+d, a:a+c]
		faceCoords = [a, b, c, d]
	return img, faceBox, faceCoords


if __name__ == "__main__":
	files = os.listdir('exampleFace')
	images = [file for file in files if 'jpg' in file]
	for image in images:
		img = cv2.imread('exampleFace/' + image)
		detected_faces, _, _ = detectFaces(img)
		cv2.imwrite('exampleFace/detected_faces/' + image, detected_faces)

