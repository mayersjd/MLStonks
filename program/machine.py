import tensorflow as tf
from program import dataset
from models import mlp as mlp


def go(inputSize, forecast, stocksToRead, trainingFraction, saveWeights, saveName, loadWeights, loadName):
    # desiredInputSize: desired number of training inputs, 1 for day, 5 for week, 20 for month, 240 for year.
    # desiredPredictionRange: this changes the labels for training/validation
    # Both of these sort of assume that data is available for every business day, which is not always a valid assumption

    # This chunk of code is necessary to avoid an issue with CUDA
    # Copied from" https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
        try:
            # Change this limit to make sure you don't exceed available memory on your GPU
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    (trainData, trainLabels, testData, testLabels) = dataset.formatData(inputSize=inputSize, forecast=forecast, stocksToRead=stocksToRead, trainingFraction=trainingFraction)

    mlp.network(trainData=trainData, trainLabels=trainLabels, testData=testData, testLabels=testLabels, inputs=inputSize, saveWeights=saveWeights, saveName=saveName, loadWeights=loadWeights, loadName=loadName)

    return print("Stonks go up!")


