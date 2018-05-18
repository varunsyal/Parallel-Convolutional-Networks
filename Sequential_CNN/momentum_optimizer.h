#ifndef __MOMENTUM_H_INCLUDED__   
#define __MOMENTUM_H_INCLUDED__  

#include<math.h>
#include "optimizer.h"

using namespace std;

template <class T>
class MomentumOptimizer: public Optimizer<T> {
public:
	float MOMENTUM = 0.6;
	MomentumOptimizer(float momentum): MOMENTUM(momentum) {
		(void)0;
	}

	void updateWeight(T& weight, Gradient<T> grad, float learningRate) {

		float m = (grad.value + grad.prevValue * MOMENTUM);
		weight -= learningRate  * m + learningRate * (1.0 - m) * weight;
	}

	void updateGradient(Gradient<T>& grad) {
		grad.prevValue = grad.prevValue + grad.value * MOMENTUM;
	}

};

#endif