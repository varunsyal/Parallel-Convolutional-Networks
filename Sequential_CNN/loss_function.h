#ifndef __LOSSFUNCTION_H_INCLUDED__   
#define __LOSSFUNCTION_H_INCLUDED__  

#include<math.h>
#include<typeinfo>

using namespace std;

class LossFunction {
public:
	void operator() (Tensor<float> output, Tensor<float>target, float &lossValue, Tensor<float> &outGrads)	{
		// cerr << "OUTPUT:\n";
		// output.printTensor();
		// cerr << "TARGET:\n";
		// target.printTensor();
		lossValue = calculateLoss(output, target);
		// cerr << "LOSS= " << lossValue << endl;
		outGrads = calculateOutGrads(output, target);
	}

	virtual float calculateLoss(Tensor<float> output, Tensor<float>target) = 0;

	virtual Tensor<float> calculateOutGrads(Tensor<float> output, Tensor<float>target)  = 0;

};

#endif