#ifndef __LAYER_H_INCLUDED__   
#define __LAYER_H_INCLUDED__  

#include "tensor.h"
#include "optimizer.h"

using namespace std;

enum Type {conv_layer, pool_layer, relu_layer, fc_layer, activation_layer};

class Layer {
public:
	Tensor<float> *in;
	Tensor<float> *out;
	Tensor<float> *gradIn;
	Type type;

	virtual Tensor<float>& operator() (Tensor<float> input) = 0;

	virtual void calculateGradients(Tensor<float> gradOut) = 0;

	virtual void updateWeights(Optimizer<float>* optimizer, float learningRate) = 0;

};

#endif