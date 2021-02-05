import os
from pathlib import Path
import glob


def formatFiles():
    currentDirectory = os.getcwd()  # Get the current working directory

    stockDataPath = os.path.join(currentDirectory, r'historicaldata\stocks')    # path to historical data for stocks
    ETFDataPath = os.path.join(currentDirectory, r'historicaldata\ETFs')    # path to historical data for ETFs
    stockFiles = Path(stockDataPath).glob('*.txt')  # make a list of all the .txt files
    ETFFiles = Path(ETFDataPath).glob('*.txt')  # make a list of all the .txt files

    # Rename all the .txt files to .csv
    for file in stockFiles:
        file.replace(file.with_suffix('.csv'))
    for file in ETFFiles:
        file.replace(file.with_suffix('.csv'))


def formatData():

    return
