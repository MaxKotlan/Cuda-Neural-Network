objects = main.o idx.o idx-cuda.o neuralnetwork.o layer.o matrix.o normalizebyte.o image-cuda.o

all: $(objects)
	nvcc $(objects) -o NeuralNetwork -lcublas

%.o: %.cpp
	nvcc -x cu -I ../include/ -dc $< -o $@

clean:
	rm -f *.o app