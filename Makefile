
BIN=./bin/
SOURCE=./src/
TESTDIR=./test/
MAIN = main.obj idx.obj idx-cuda.obj neuralnetwork.obj layer.obj matrix.obj normalizebyte.obj image-cuda.obj
TEST = test.obj matrix-test.obj
MAINOBJ = $(addprefix $(SOURCE),$(MAIN))
TESTOBJ = $(addprefix $(TESTDIR),$(TEST))

all: $(MAINOBJ)
	nvcc $(MAINOBJ) -o NeuralNetwork -lcublas

test: $(TESTOBJ)
	nvcc $(TESTOBJ) -o NeuralNetworkTests -lcublas -lNeuralNetwork

%.obj: %.cpp
	nvcc -x cu -I ./include/ -I ./ -dc $< -o $@ 

clean:
	rm -f *.obj NeuralNetwork NeuralNetworkTests