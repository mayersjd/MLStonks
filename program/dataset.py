from pathlib import Path
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime as dt
import edgar as ed
import csv


def importData():
    currentDirectory = os.getcwd()  # Get the current working directory
    tickerList = []
    tickerListFile = os.path.join(currentDirectory, r'historicaldata\All_Tickers_and_Sectors.csv')  # path to historical data for stocks
    """with open(tickerListFile, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for index, lines in enumerate(csv_reader):
            if index != 0:
                tickerList.append(lines[0])"""

    #print(type(str(tickerList[0])), type(tickerList[0]))
    #edgar = ed.Edgar()
    #possible_companies = edgar.find_company_name(tickerList[0])
    #print(possible_companies)
    #company = ed.Company("INTERNATIONAL BUSINESS MACHINES CORP", "0000051143")
    #doc = company.get_10K()
    #text = ed.TXTML.parse_full_10K(doc)


    # company = ed.Company("Oracle Corp", "0001341439")
    # tree = company.get_all_filings(filing_type="10-K")
    # docs = ed.Company.get_documents(tree)


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


def formatData(inputSize, forecast, stocksToRead, trainingFraction):
    print('Reading in and formatting data files for training and testing...')
    # Read in the data set for training
    currentDirectory = os.getcwd()  # Get the current working directory
    stockDataPath = os.path.join(currentDirectory, r'historicaldata\stocks')
    stockFiles = Path(stockDataPath).glob('*.csv')  # make a generator of all the .csv files

    count = 0
    labeledData = []
    for file in stockFiles:
        if count > stocksToRead:
            break
        else:
            if os.stat(file).st_size == 0:  # Check to make sure the file has content
                continue
            else:
                data = pd.read_csv(file, parse_dates=['Date'])     # Read in the data for a stock, and change the date information to datetime objects
                data.pop('OpenInt')     # Remove the 'OpenInt' column, as it appears to be all zeroes all the time

                # Convert the values in the Date column to integers referencing the number of days passed since 1900-1-1
                for i in range(len(data)):
                    data['Date'][i] = float((data['Date'][i] - dt.datetime(1900, 1, 1)).days)   # This raises an error, but I don't think it's a problem
                data['Volume'] = data['Volume'].astype(float)

                # Splitting the dataframe up into chunks of the appropriate input size
                if len(data) < inputSize:   # Check the size of the input file to make sure it has enough rows for the desired input size
                    continue
                else:
                    split_remainder = len(data) % inputSize  # Check if the current size of the dataframe is divided evenly by the desired input size
                    data.drop(data.tail(split_remainder).index, inplace=True)   # reshape the dataframe by dropping the remainder
                    splitSize = len(data) / inputSize
                    dataSplit = np.array_split(data, splitSize)     # split the data frame into equally sized chunks

                    # Call function to create the labels for training/validation
                    labeledData += (createLabels(data=data, dataSplit=dataSplit, forecast=forecast))
        count += 1

    # Call function to shuffle the data and split into testing and training sets
    trainData, trainLabels, testData, testLabels = shuffleData(labeledData=labeledData, trainingFraction=trainingFraction)

    print('Done!')
    return trainData, trainLabels, testData, testLabels


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


def shuffleData(labeledData, trainingFraction):
    # trainingFraction: the division of the data into training and validation sets

    # Shuffle the data, then split it into training and validation sets
    np.random.shuffle(labeledData)
    trainingDataLength = round(len(labeledData) * trainingFraction)
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
