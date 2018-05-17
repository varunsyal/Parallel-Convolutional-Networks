#ifndef __ADAGRAD_H_INCLUDED__   
#define __ADAGRAD_H_INCLUDED__  

#include<math.h>
#include "optimizer.h"
using namespace std;

template <class T>
class AdagradOptimizer: public Optimizer<T> {
public:

	void updateWeight(T& weight, Gradient<T> grad, float learningRate) {
		weight = weight - learningRate / sqrt(grad.prevValue) * grad.value;
	}

	void updateGradient(Gradient<T>& grad) {
		grad.prevValue = grad.prevValue + grad.value * grad.value;
	}

};

#endif