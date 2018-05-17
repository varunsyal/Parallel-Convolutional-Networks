#ifndef __ACT_LAYER_H_INCLUDED__   
#define __ACT_LAYER_H_INCLUDED__  

#include<cstdlib> 
#include "layer.h"
#include "gradient.h"
#include "adagrad_optimizer.h"
#include<vector>
#include<utility>
#define INF 1999999999

using namespace std;

class ActivationLayer: public Layer {
public:
	TensorSize inSize;

	ActivationLayer(TensorSize inSize_ ): inSize(inSize_) {
			in = new Tensor<float> (inSize_.x, inSize_.y, inSize_.z);
			out = new Tensor<float> (inSize_.x, inSize_.y, inSize_.z);
			gradIn = new Tensor<float> (inSize);
	}


	Tensor<float>& operator() (Tensor<float> input) {
		*in = input;
		this->forward();
		return *out;
	}

	virtual float _activate_(float in) = 0;
	virtual float _gradient_(float in) = 0;

	//Forward Convolution Pass
	void forward() {
		for (int x = 0; x < inSize.x; x++) {
			for (int y = 0; y < inSize.y; y++) {
				for (int z = 0; z < inSize.z; z++) {
					out->get(x,y,z) = _activate_(in->get(x,y,z));
				}
			}
		}
	}


	//Calculating gradients for Backpropogation
	void calculateGradients(Tensor<float> gradOut) {
		assert(gradOut.tsize.x == out->tsize.x);
		assert(gradOut.tsize.y == out->tsize.y);
		assert(gradOut.tsize.z == out->tsize.z);
		
		for (int x = 0; x < inSize.x; x++) {
			for (int y = 0; y < inSize.y; y++) {
				for (int z = 0; z < inSize.z; z++) {
					gradIn->get(x,y,z) = 0;
				}
			}
		}

		for (int x = 0; x < inSize.x; x++) {
			for (int y = 0; y < inSize.y; y++) {
				for (int z = 0; z < inSize.z; z++) {
					gradIn->get(x,y,z) = gradOut(x,y,z) * _gradient_(in->get(x,y,z));
				}
			}
		}
	}


	void updateWeights(Optimizer<float> optimizer, float learningRate) {
		(void)0;
	}


};

#endif