A C++ Library for Convolutional Neural Nets with Parallel Computing(openMP, CUDA, MPI)

## Usage: ##
g++ -std=c++11 lenet.cpp -o lenet
./lenet

This is an example network derived from Yann LeCun's LeNet.

## Create your own Network ##

* You can create your own deep neural network class by deriving from the _Model_ class and adding all your layers in order by using _addLayer()_ method.
* You can also introduce your own Activation layers by extending the _ActivationLayer_.
* You can create your custom Loss functions by extending the _LossFunction_ class.

## Work in Progress ##
Optimizations using openMP, CUDA, MPI
