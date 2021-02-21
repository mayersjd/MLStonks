import tensorflow as tf
import os
import numpy as np


def network(trainData, trainLabels, testData, testLabels, inputs, saveWeights, saveName, loadWeights, loadName):
    print('Building and training Convolutional Long-Short Term Memory')
    # Neural network model: Sequential Convolutional Long-Short Term Memory
    #print(None, inputs, len(trainData[0][0]), 1)
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=[None, 1, len(trainData[0][0]), 1]),
        tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.input_shape)

    # Reshaping the training and testing data sets to appropriately be read by the model
    trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1], trainData.shape[2], 1))
    testData = np.reshape(testData, (testData.shape[0], testData.shape[1], testData.shape[2], 1))

    if saveWeights:     # Train the model and save the weights
        model.fit(trainData, trainLabels, epochs=10, validation_data=(testData, testLabels))
        currentDirectory = os.getcwd()  # Get the current working directory
        model.save_weights(filepath=os.path.join(currentDirectory, r'models\weights\convlstm\{}'.format(saveName)))
    elif loadWeights:   # Load the previously saved weights and evaluate the model on the (new) test set
        currentDirectory = os.getcwd()  # Get the current working directory
        model.load_weights(filepath=os.path.join(currentDirectory, r'models\weights\convlstm\{}'.format(loadName)))
        model.evaluate(testData, testLabels, verbose=2)
    else:   # If not saving or loading, train a whole new model and evaluate it
        model.fit(trainData, trainLabels, epochs=10, validation_data=(testData, testLabels))

