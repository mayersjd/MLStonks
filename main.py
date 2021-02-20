import argparse
from program import machine
from program import dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A machine learning program to evaluate stocks')
    parser.add_argument("--inputSize",
                        help="Specifies how many inputs (neurons) should be used for the first layer of the neural network. Default is 240 days (i.e. 1 year's worth of historical data).",
                        default=240, type=int)
    parser.add_argument("--forecast",
                        help="Specifies how many (business) days into the future the network should try to predict a value. Default is 20 days (1 month).",
                        default=20, type=int)
    parser.add_argument("--stocksToRead",
                        help="Specifies how many stocks' historical data should be used to train/validate the model. Default is 10.",
                        default=10, type=int)
    parser.add_argument("--trainingFraction",
                        help="Specifies what percentage of the data you want to be set aside for training. Default is 0.6.",
                        default=0.6, type=float)
    parser.add_argument("--saveWeights",
                        help="Used to save trained model weights, default is False",
                        default=False, type=bool)
    parser.add_argument("--saveName",
                        help="Name used to save model weights, only used when saveWeights is True. You must save weights using file extension '.h5'",
                        default="", type=str)
    parser.add_argument("--loadWeights",
                        help="Used to save trained model weights, default is False",
                        default=False, type=bool)
    parser.add_argument("--loadName",
                        help="Name used to load model weights, only used when loadWeights is True. You must save weights using file extension '.h5'",
                        default="", type=str)
    args = parser.parse_args()
    dataset.importData()
    machine.go(inputSize=args.inputSize, forecast=args.forecast, stocksToRead=args.stocksToRead, trainingFraction=args.trainingFraction, saveWeights=args.saveWeights, saveName=args.saveName, loadWeights=args.loadWeights, loadName=args.loadName)

