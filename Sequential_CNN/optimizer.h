#ifndef __OPTIMIZER_H_INCLUDED__   
#define __OPTIMIZER_H_INCLUDED__  

#include "gradient.h"

using namespace std;

template <class T>
class Optimizer {
public:
	virtual void updateWeight(T& weight, Gradient<T> grad, float learningRate) = 0;
	
	virtual void updateGradient(Gradient<T>& grad) = 0;
};

#endif