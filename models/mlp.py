import tensorflow as tf
import os


def network(trainData, trainLabels, testData, testLabels, inputs, saveWeights, saveName, loadWeights, loadName):
    print('Building and training Multi Layer Perceptron')
    # Neural network model: Sequential Multi-Layer Perceptron
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(inputs, len(trainData[0][0]))),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])

    if saveWeights:     # Train the model and save the weights
        model.fit(trainData, trainLabels, epochs=100, validation_data=(testData, testLabels))
        currentDirectory = os.getcwd()  # Get the current working directory
        model.save_weights(filepath=os.path.join(currentDirectory, r'models\weights\mlp\{}'.format(saveName)))
    elif loadWeights:   # Load the previously saved weights and evaluate the model on the (new) test set
        currentDirectory = os.getcwd()  # Get the current working directory
        model.load_weights(filepath=os.path.join(currentDirectory, r'models\weights\mlp\{}'.format(loadName)))
        model.evaluate(testData, testLabels, verbose=2)
    else:   # If not saving or loading, train a whole new model and evaluate it
        model.fit(trainData, trainLabels, epochs=100, validation_data=(testData, testLabels))

