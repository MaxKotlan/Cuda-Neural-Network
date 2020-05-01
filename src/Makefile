all:
	nvcc -o ConvolutionalNeuralNetwork -lcudart -lcublas -I ../lib/ -I ../include/ main.cu idx.cpp idx-cuda.cpp neuralnetwork.cu layer.cu matrix.cu
