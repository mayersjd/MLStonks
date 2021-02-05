from pathlib import Path
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime as dt


def formatFiles():
    currentDirectory = os.getcwd()  # Get the current working directory

    stockDataPath = os.path.join(currentDirectory, r'historicaldata\stocks')    # path to historical data for stocks
    ETFDataPath = os.path.join(currentDirectory, r'historicaldata\ETFs')    # path to historical data for ETFs
    stockFiles = Path(stockDataPath).glob('*.txt')  # make a generator of all the .txt files
    ETFFiles = Path(ETFDataPath).glob('*.txt')  # make a generator of all the .txt files

    # Rename all the .txt files to .csv
    for file in stockFiles:
        file.replace(file.with_suffix('.csv'))
    for file in ETFFiles:
        file.replace(file.with_suffix('.csv'))


def formatData(inputSize, forecast):
    print('Reading in and formatting data files for training and testing...')
    # Read in the data set for training
    currentDirectory = os.getcwd()  # Get the current working directory
    stockDataPath = os.path.join(currentDirectory, r'historicaldata\stocks')
    stockFiles = Path(stockDataPath).glob('*.csv')  # make a generator of all the .csv files

    count = 0
    labeledData = []
    for file in stockFiles:
        if count > 0:
            break
        else:
            if os.stat(file).st_size == 0:
                continue
            else:
                data = pd.read_csv(file, parse_dates=['Date'])     # Read in the data for a stock, and change the date information to datetime objects
                data.pop('OpenInt')     # Remove the 'OpenInt' column, as it appears to be all zeroes all the time

                # Convert the values in the Date column to integers referencing the number of days passed since 1900-1-1
                for i in range(len(data)):
                    data['Date'][i] = float((data['Date'][i] - dt.datetime(1900, 1, 1)).days)   # This raises an error, but I don't think it's a problem
                data['Volume'] = data['Volume'].astype(float)

                # Splitting the dataframe up into chunks of the appropriate input size
                split_remainder = len(data) % inputSize  # Check if the current size of the dataframe is divided evenly by the desired input size
                data.drop(data.tail(split_remainder).index, inplace=True)   # reshape the dataframe by dropping the remainder
                splitSize = len(data) / inputSize
                dataSplit = np.array_split(data, splitSize)     # split the data frame into equally sized chunks

                # Call function to create the labels for training/validation
                labeledData += (createLabels(data, dataSplit, forecast))
        count += 1

    # Call function to shuffle the data and split into testing and training sets
    trainingSet, trainingLabels, testingSet, testingLabels = shuffleData(labeledData)

    print('Done!')
    return trainingSet, trainingLabels, testingSet, testingLabels


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

def shuffleData(labeledData):
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

    return trainingSet, trainingLabels, testingSet, testingLabels
