import tensorflow as tf
from program import dataset


def go():
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

    # Both of these sort of assume that data is available for every business day, which is not always a valid assumption
    # This influences the size of the input to the NN
    desiredInputSize = 120  # desired number of training inputs, 1 for day, 5 for week, 20 for month, 240 for year.
    # This changes the labels for training/validation
    desiredPredictionRange = 20  # desired timeframe for inferencing, 1 for day, 5 for week, 20 for month, 240 for year.

    (trainingSet, trainingLabels, testingSet, testingLabels) = dataset.formatData(desiredInputSize, desiredPredictionRange)

    multiLayerPerceptron(trainingSet, trainingLabels, testingSet, testingLabels, desiredInputSize)

    return print("Stonks go up!")

def multiLayerPerceptron(trainData, trainLabels, testData, testLabels, inputs):
    print('Building and training multi layer perceptron')
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
