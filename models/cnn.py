import tensorflow as tf
import os
import numpy as np

def network(trainData, trainLabels, testData, testLabels, inputs, saveWeights, saveName, loadWeights, loadName):
    print('Building and training Convolutional Neural Network')
    # Neural network model: Sequential CNN
    input_shape = [4, 1, len(trainData[0]), len(trainData[0][0])]  # Shape should be [batch size, channels, rows, cols]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, padding='same', activation='relu', input_shape=input_shape[1:], data_format="channels_first"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, 'softmax')
    ])
    model.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])

    # Reshaping the training and testing data sets to appropriately be read by the model (total number of inputs, 1 channel, number of rows, number of columns)
    trainData = np.reshape(trainData, (trainData.shape[0], 1, trainData.shape[1], trainData.shape[2]))
    testData = np.reshape(testData, (testData.shape[0], 1, testData.shape[1], testData.shape[2]))

    if saveWeights:     # Train the model and save the weights
        model.fit(trainData, trainLabels, epochs=100, validation_data=(testData, testLabels))
        currentDirectory = os.getcwd()  # Get the current working directory
        model.save_weights(filepath=os.path.join(currentDirectory, r'models\weights\cnn\{}'.format(saveName)))
    elif loadWeights:   # Load the previously saved weights and evaluate the model on the (new) test set
        currentDirectory = os.getcwd()  # Get the current working directory
        model.load_weights(filepath=os.path.join(currentDirectory, r'models\weights\cnn\{}'.format(loadName)))
        model.evaluate(testData, testLabels, verbose=2)
    else:   # If not saving or loading, train a whole new model and evaluate it
        model.fit(trainData, trainLabels, epochs=100, validation_data=(testData, testLabels))
