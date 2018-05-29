#ifndef __LAYER_H_INCLUDED__   
#define __LAYER_H_INCLUDED__  

#include "tensor.h"
#include "optimizer.h"
#include <omp.h>
#include <vector>

using namespace std;

enum Type {conv_layer, pool_layer, relu_layer, fc_layer, activation_layer};

class Layer {
public:
	Tensor<float> *in;
	Tensor<float> *out;
	Tensor<float> *gradIn;
	Type type;
	omp_lock_t writelock;
	int indx;


	virtual Tensor<float> operator() (Tensor<float> input, vector<Tensor<float> >& layer_ins) = 0;

	virtual Tensor<float> calculateGradients(Tensor<float> gradOut, vector<Tensor<float> > layer_ins) = 0;

	virtual void updateWeights(Optimizer<float>* optimizer, float learningRate, int batchSize) = 0;

	virtual void clearGradients() = 0;

	virtual void printWeights() = 0;

};

#endif