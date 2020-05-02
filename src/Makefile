objects = main.obj idx.obj idx-cuda.obj neuralnetwork.obj layer.obj matrix.obj normalizebyte.obj image-cuda.obj

all: $(objects)
	nvcc $(objects) -o NeuralNetwork -lcublas

%.obj: %.cpp
	nvcc -x cu -I ../include/ -dc $< -o $@

clean:
	rm -f *.obj app