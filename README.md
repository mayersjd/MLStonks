# MLStocks
Training neural networks to buy and sell stocks

This was originally conceieved as a project for UC's Revolution Hackathon. The project was so exciting though, that we decided to work on it prior to the hackathon, and so will not be "competing" for any prizes, but will still be making use of the time to continue development.

See requirements.txt for required python packages. In order to use CUDA you should have an appropriate GPU and the following versions installed. [This](https://www.tensorflow.org/install/gpu) guide should help. You will need to sign up for the NVIDIA Developer program (it's free).
1. Python 3.8.x
2. TensorFlow 2.4.1
3. [CUDA 11.0.3](https://developer.nvidia.com/cuda-11.0-update1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
4. [CUDNN 8.0.4.30 for CUDA 11.0](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse804-110)

Usage: 
main.py [-h] [--inputSize INPUTSIZE] [--forecast FORECAST] [--stocksToRead STOCKSTOREAD] [--trainingFraction TRAININGFRACTION]

A machine learning program to evaluate stocks

optional arguments:
  -h, --help

		show this help message and exit
  
  --inputSize INPUTSIZE
  
  		Specifies how many inputs (neurons) should be used for the first layer of the neural network.
		Default is 240 days (i.e. 1 year's worth of historical data).
  
  --forecast FORECAST
	
		Specifies how many (business) days into the future the network should try to predict a value.
		Default is 20 days (1 month).
  
  --stocksToRead STOCKSTOREAD
	
		Specifies how many stocks' historical data should be used to train/validate the model.
		Default is 10.
  
  --trainingFraction TRAININGFRACTION
	
		Specifies what percentage of the data you want to be set aside for training.
		Default is 0.6.
