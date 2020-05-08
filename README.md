
# Cuda Neural Network 

My implementation of  a Neural Network. Currently uses Sigmoid activation function. (others will be supported soon) and gradient descent for tuning the network.

## Building
[Install  NVIDIA's Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)
Install make ( I installed it with a package manager [scoop](https://scoop.sh/) )
clone repo.

    cd Cuda-Neural-Network
    make


## Interface

This is my implementation of a standard Neural Network. It uses gradient descent for tuning the network. 
To import the MNIST database, simply create the following:
Stored in host ram:

    IDX::ImageDatabase t10k("data/t10k-images.idx3-ubyte");


Stored in vram:

    IDX::CudaImageDatabase t10k("data/t10k-images.idx3-ubyte");

To create a Neural Network. Invoke this constructor:

    NeuralNetwork mynn( inputLayerSize, hiddenLayerSize, hiddenLayerCount, outputLayerSize, learrningRate );

To train data:

    IDX::ImageDatabase t10ktrain("data/train-images.idx3-ubyte");
    IDX::LabelDatabase t10ktrainlab("data/train-labels.idx1-ubyte");

	for (int i = 0; i < 10ktrain.size(); i++){
	    std::vector<float> normalizedimage = t10ktrain.GetImage(i).Normalize();
	    uint32_t label = t10ktrainlab.GetLabel(i);
	    mynn.TrainSingle(normalizedimage , label);
    }

To forward propagate data after training:

    IDX::ImageDatabase t10k("data/t10k-images.idx3-ubyte");
    std::vector<float> normalizedimage = t10k.GetImage(0).Normalize();
    mynn.ForwardPropagate(normalizedimage);
