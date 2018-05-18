#ifndef __CROSSENTROPY_LOSS_H_INCLUDED__   
#define __CROSSENTROPY_LOSS_H_INCLUDED__  

#include "loss_function.h"
using namespace std;

class SoftmaxCrossEntropy: public LossFunction {
	
	float calculateLoss(Tensor<float> output, Tensor<float>target) {
		assert(output.tsize.x == target.tsize.x);
		assert(output.tsize.y == target.tsize.y == 1);
		assert(output.tsize.z == target.tsize.z == 1);

		Tensor<float> softmaxOutput(output.tsize);
		float den = 0.0, loss = 0.0;
		for (int i = 0; i < output.tsize.x; i++) {
			softmaxOutput(i,0,0) = exp(output(i,0,0));
			den += softmaxOutput(i,0,0);
		}

		for (int i = 0; i < output.tsize.x; i++) {
			softmaxOutput(i,0,0) /= den;
		}

		for (int i = 0; i < output.tsize.x; i++) {
			loss -= target(i,0,0) * log(softmaxOutput(i,0,0));
		}
		return loss;
	}

	Tensor<float> calculateOutGrads(Tensor<float> output, Tensor<float>target) {
		assert(output.tsize.x == target.tsize.x);
		assert(output.tsize.y == target.tsize.y == 1);
		assert(output.tsize.z == target.tsize.z == 1);

		Tensor<float> softmaxOutput(output.tsize);
		float den = 0.0, loss = 0.0;
		for (int i = 0; i < output.tsize.x; i++) {
			softmaxOutput(i,0,0) = exp(output(i,0,0));
			den += softmaxOutput(i,0,0);
		}

		for (int i = 0; i < output.tsize.x; i++) {
			softmaxOutput(i,0,0) /= den;
		}

		return (softmaxOutput - target);
	}
};

#endif