#ifndef __SGD_H_INCLUDED__   
#define __SGD_H_INCLUDED__  

#include<math.h>
#include "optimizer.h"
using namespace std;

template <class T>
class SGDOptimizer: public Optimizer<T> {
public:

	void updateWeight(T& weight, Gradient<T> grad, float learningRate) {
		weight = weight - learningRate * grad.value;
	}

	void updateGradient(Gradient<T>& grad) {
		(void)0;
	}

};

#endif