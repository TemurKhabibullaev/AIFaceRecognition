from keras.layers import Input, Lambda
from keras.models import Model
import numpy as np
import functionsNN


# Creating Seamese Neural Networks & Importing Training and Testing Results
faces_dir = 'facesDatabase/'
(XTrain, YTrain), (XTest, YTest) = functionsNN.get_data(faces_dir)
numClasses = len(np.unique(YTrain))
inputShape = XTrain.shape[1:]
inputTop = Input(shape=inputShape)
inputBottom = Input(shape=inputShape)
sharedNetwork = functionsNN.sharedNetwork(inputShape)
outputTop = sharedNetwork(inputTop)
outputBottom = sharedNetwork(inputBottom)
distance = Lambda(functionsNN.euclideanDistance, output_shape=(1,))([outputTop, outputBottom])
model = Model(inputs=[inputTop, inputBottom], outputs=distance)

# Training:
trainingPairs, trainingLabels = functionsNN.createdPairs(XTrain, YTrain, numClasses=numClasses)
model.compile(loss=functionsNN.contrastiveLoss, optimizer='adam', metrics=[functionsNN.accuracy])
model.fit([trainingPairs[:, 0], trainingPairs[:, 1]], trainingLabels, batch_size=128, epochs=10)
model.save('siameseModel.h5')
