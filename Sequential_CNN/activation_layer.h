#ifndef __ACT_LAYER_H_INCLUDED__   
#define __ACT_LAYER_H_INCLUDED__  

#include <ctime>
#include <cstdlib> 
#include "layer.h"
#include "gradient.h"
#include "adagrad_optimizer.h"
#include <vector>
#include <utility>
#include <omp.h>
#define INF 1999999999

using namespace std;

class ActivationLayer: public Layer {
public:
	TensorSize inSize;

	ActivationLayer(int indx_, TensorSize inSize_ ): inSize(inSize_) {
			this->indx = indx_;
			this->type = activation_layer;
			in = new Tensor<float> (inSize_.x, inSize_.y, inSize_.z);
			out = new Tensor<float> (inSize_.x, inSize_.y, inSize_.z);
			gradIn = new Tensor<float> (inSize);
	}


	Tensor<float> operator() (Tensor<float> input, vector<Tensor<float> > &layer_ins) {
		Tensor<float> *in_ = new Tensor<float> (inSize.x, inSize.y, inSize.z);
		*in_ = input;
		layer_ins.push_back(*in_);
		return this->forward(in_);
		
	}

	virtual float _activate_(float in) = 0;
	virtual float _gradient_(float in) = 0;

	//Forward Convolution Pass
	Tensor<float> forward(Tensor<float> *in_) {
		Tensor<float> *out_ = new Tensor<float> (inSize.x, inSize.y, inSize.z);
		for (int x = 0; x < inSize.x; x++) {
			for (int y = 0; y < inSize.y; y++) {
				for (int z = 0; z < inSize.z; z++) {
					out_->get(x,y,z) = _activate_(in_->get(x,y,z));
				}
			}
		}
		return *out_;
	}


	//Calculating gradients for Backpropogation
	Tensor<float> calculateGradients(Tensor<float> gradOut, vector<Tensor<float> > layer_ins) {
		Tensor<float> *gradIn_ = new Tensor<float> (inSize);

		assert(gradOut.tsize.x == out->tsize.x);
		assert(gradOut.tsize.y == out->tsize.y);
		assert(gradOut.tsize.z == out->tsize.z);
		
		for (int x = 0; x < inSize.x; x++) {
			for (int y = 0; y < inSize.y; y++) {
				for (int z = 0; z < inSize.z; z++) {
					gradIn_->get(x,y,z) = 0;
				}
			}
		}

		for (int x = 0; x < inSize.x; x++) {
			for (int y = 0; y < inSize.y; y++) {
				for (int z = 0; z < inSize.z; z++) {
					gradIn_->get(x,y,z) = gradOut(x,y,z) * _gradient_(layer_ins[this->indx].get(x,y,z));
				}
			}
		}
		
		return *gradIn_;
	}


	void updateWeights(Optimizer<float> *optimizer, float learningRate, int batchSize) {
		(void)0;
	}

	void clearGradients() {
		(void)0;
	}

	void printWeights() {
		(void)0;
	}

	void addGlobalGradients() {
		(void)0;
	}
};

#endif