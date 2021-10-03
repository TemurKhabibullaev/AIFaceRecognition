#Temur Khabibullaev
# Here I train Siamese Neural Networks
import tools
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model

faces_dir = 'humanFaces/'

# Import Training and Testing Data
(X_train, Y_train), (X_test, Y_test) = tools.get_data(faces_dir)
num_classes = len(np.unique(Y_train))

# Create Siamese Neural Network
input_shape = X_train.shape[1:]
shared_network = tools.create_shared_network(input_shape)
input_top = Input(shape=input_shape)
input_bottom = Input(shape=input_shape)
output_top = shared_network(input_top)
output_bottom = shared_network(input_bottom)
distance = Lambda(tools.euclidean_distance, output_shape=(1,))([output_top, output_bottom])
model = Model(inputs=[input_top, input_bottom], outputs=distance)

# Train the model
training_pairs, training_labels = tools.create_pairs(X_train, Y_train, num_classes=num_classes)
model.compile(loss=tools.contrastive_loss, optimizer='adam', metrics=[tools.accuracy])
model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels, batch_size=128, epochs=10)

# Save the model
model.save('siamese_nn.h5')