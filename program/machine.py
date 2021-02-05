import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime as dt


def go():
    # This chunk of code is necessary to avoid an issue with CUDA
    # Copied from" https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            # Change this limit to make sure you don't exceed available memory on your GPU
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # Both of these sort of assume that data is available for every business day, which is not always a valid assumption
    # This influences the size of the input to the NN
    desiredInputSize = 20  # desired number of training inputs, 1 for day, 5 for week, 20 for month, 240 for year.
    # This changes the labels for training/validation
    desiredPredictionRange = 20  # desired timeframe for inferencing, 1 for day, 5 for week, 20 for month, 240 for year.

    # Read in the data set for training
    currentDirectory = os.getcwd()  # Get the current working directory
    stockDataPath = os.path.join(currentDirectory, r'historicaldata\stocks\a.us.csv')
    data = pd.read_csv(stockDataPath, parse_dates=['Date']) # Read in the data for a stock, and change the date information to datetime objects
    data.pop('OpenInt')     # Remove the 'OpenInt' column, as it appears to be all zeroes all the time

    # Convert the values in the Date column to integers referencing the number of days passed since 1900-1-1
    for i in range(len(data)):
        data['Date'][i] = float((data['Date'][i] - dt.datetime(1900, 1, 1)).days)   # This raises an error, but I don't think it's a problem
    data['Volume'] = data['Volume'].astype(float)

    # Splitting the dataframe up into chunks of the appropriate input size
    split_remainder = len(data) % desiredInputSize  # Check if the current size of the dataframe is divided evenly by the desired input size
    data.drop(data.tail(split_remainder).index, inplace=True)   # reshape the dataframe by dropping the remainder
    splitSize = len(data) / desiredInputSize
    dataSplit = np.array_split(data, splitSize)     # split the data frame into equally sized chunks

    # Call function to create the labels for training/validation
    labeledData = createLabels(data, dataSplit, desiredPredictionRange)
    #print(labeledData[0])

    # Shuffle the data, then split it into training and validation sets
    np.random.shuffle(labeledData)
    trainingDataLength = round(len(labeledData) * 0.7)
    trainingSet = []
    trainingLabels = []
    testingSet = []
    testingLabels = []
    for i in range(len(labeledData)):
        if i <= trainingDataLength:
            trainingSet.append(labeledData[i][0])
            trainingLabels.append(labeledData[i][1])
        else:
            testingSet.append(labeledData[i][0])
            testingLabels.append(labeledData[i][1])
    # We have to convert the list of arrays to a multidimensional numpy array and then convert it into a tensor object
    trainingSet = np.asarray(trainingSet).astype(np.float32)
    trainingSet = tf.convert_to_tensor(trainingSet)   # Converting the training set to a tensor
    trainingLabels = np.asarray(trainingLabels).astype(np.uint8)    # Converting the labels to a numpy array
    testingSet = np.asarray(testingSet).astype(np.float32)
    testingSet = tf.convert_to_tensor(testingSet)   # Converting the testing set to a tensor
    testingLabels = np.asarray(testingLabels).astype(np.uint8)    # Converting the labels to a numpy array

    multiLayerPerceptron(trainingSet, trainingLabels, testingSet, testingLabels, desiredInputSize)

    return print("Stonks go up!")


def createLabels(data, dataSplit, forecast):
    labels = []
    for i, split in enumerate(dataSplit):
        futureIndex = i * len(split) + forecast
        if futureIndex >= len(data):
            # dataSplit = dataSplit[:-(len(dataSplit)-i)]
            break
        else:
            currentValueAvg = (split['Open'][i * len(split)] + split['High'][i * len(split)] + split['Low'][i * len(split)] + split['Close'][i * len(split)]) / 4
            futureValueAvg = (data['Open'][futureIndex] + data['High'][futureIndex] + data['Low'][futureIndex] + data['Close'][futureIndex]) / 4
            if futureValueAvg > currentValueAvg:
                labels.append([split.values, 1])    # Label of 1 indicates the stock rises in price over the desired timeframe
            else:
                labels.append([split.values, 0])    # Label of 0 indicates the stock falls in price over the desired timeframe
    return labels


def multiLayerPerceptron(trainData, trainLabels, testData, testLabels, inputs):

    # Neural network model: Sequential
    mlp = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(inputs, 6)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    mlp.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    mlp.fit(trainData, trainLabels, epochs=100)
    test_loss, test_acc = mlp.evaluate(testData, testLabels, verbose=2)
    print('\nTest accuracy:', test_acc)