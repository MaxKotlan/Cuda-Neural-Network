
BIN=./bin/
SOURCE=./src/
TESTDIR=./test/
MAIN = main.obj idx.obj idx-cuda.obj neuralnetwork.obj layer.obj matrix.obj normalizebyte.obj image-cuda.obj
TEST = test.obj

all: $(SOURCE)$(MAIN)
	nvcc $(SOURCE)$(MAIN) -o NeuralNetwork -lcublas

test: $(TESTDIR)$(TEST)
	nvcc $(TESTDIR)$(TEST) -o NeuralNetworkTests -lcublas -lNeuralNetwork.lib

%.obj: %.cpp
	nvcc -x cu -I ./include/ -I ./ -dc $< -o $@

clean:
	rm -f *.obj NeuralNetwork NeuralNetworkTests