all:
	nvcc -o ConvolutionalNeuralNetwork -I lib/ Main.cu idx.cpp
