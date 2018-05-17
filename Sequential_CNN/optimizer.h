#ifndef __OPTIMIZER_H_INCLUDED__   
#define __OPTIMIZER_H_INCLUDED__  

using namespace std;

template <class T>
class Optimizer {
public:
	void updateWeight(T& weight, Gradient<T> grad, float learningRate) {
		(void)0;
	}
	
	void updateGradient(Gradient<T>& grad) {
		(void)0;
	}
};

#endif