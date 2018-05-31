A C++ Library for Convolutional Neural Nets with Parallel Computing(openMP, CUDA, MPI)

## Usage: ##
g++ -std=c++11 -fopenmp lenet.cpp -o lenet
./lenet

* This is a multi-threaded version of the model (with data parallelism) and you can change the number of threads by using:
export OMP_NUM_THREADS=4

* For using the MPI version of code, you need to compile using mpic++:
mpic++ -std=c++11 -fopenmp lenet.cpp -o lenet

and you can run this on multi-node system!


## Create your own Network ##

* You can create your own deep neural network class by deriving from the _Model_ class and adding all your layers in order by using _addLayer()_ method.
* You can also introduce your own Activation layers by extending the _ActivationLayer_.
* You can create your custom Loss functions by extending the _LossFunction_ class.

## Work in Progress ##
Optimizations using:
openMP: COMPLETED
MPI: COMPLETED
CUDA: IN PROGRESS
