
BIN=./bin/
SOURCE=./src/
TESTDIR=./test/
MAIN = main.obj idx.obj idx-cuda.obj neuralnetwork.obj layer.obj matrix.obj normalizebyte.obj image-cuda.obj
TEST = test.obj matrix-test.obj imagedatabase-test.obj
MAINOBJ = $(addprefix $(SOURCE),$(MAIN))
MAINLIB = $(filter-out $(SOURCE)main.obj, $(MAINOBJ))
TESTOBJ = $(addprefix $(TESTDIR),$(TEST))

all: $(MAINOBJ)
	nvcc $(MAINOBJ) -o NeuralNetwork -lcublas -lcurand

lib: $(MAINLIB)
	nvcc $(MAINLIB) -o NeuralNetwork.lib -lib

test: lib $(TESTOBJ)
	nvcc $(TESTOBJ) -o NeuralNetworkTests -lcublas -lcurand --link NeuralNetwork.lib

%.obj: %.cpp
	nvcc -x cu -I ./include/ -I ./ -I ./lib/GSL/include/ -dc $< -o $@ 

clean:
	rm -f *.obj NeuralNetwork NeuralNetworkTests