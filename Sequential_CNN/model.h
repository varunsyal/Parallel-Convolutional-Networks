#ifndef __MODEL_H_INCLUDED__   
#define __MODEL_H_INCLUDED__  

#include "conv_layer.h"
#include "pool_layer.h"
#include "fc_layer.h"
#include "relu_layer.h"
#include "sigmoid_layer.h"
#include "softmax_cross_entropy.h"
#include<vector>

using namespace std;

class Model {
public:
	float learningRate;
	Optimizer<float> *optimizer;
	vector<Layer*> modelLayers;
	LossFunction *loss;

	Tensor<float> forwardPass(Tensor<float> input, bool prnt) {
		Tensor<float> x = input;
		for (int i = 0; i < modelLayers.size(); i++) {
			if (modelLayers[i]->type == fc_layer) {
				x = x.flatten();
			}
			Tensor<float> tmp = modelLayers[i]->operator()(x);
			x = tmp;
			if (prnt) {
				cout << "=======LAYER " << i << modelLayers[i]->type << "=============\n\n";
				x.printTensor();
			}
		}

		return x;
	}

	void calculateModelGradients(Tensor<float> outGrads) {
		Tensor<float> x = outGrads;
		for (int i = modelLayers.size() - 1; i >= 0; i--) {
			if (i+1 < modelLayers.size() && modelLayers[i+1] -> type == fc_layer) {
				x = x.reshape(modelLayers[i]->out->tsize);
			}
			modelLayers[i]->calculateGradients(x);
			x = *(modelLayers[i]->gradIn);
		}
	}

	void updateModelParameters() {
		for (int i = modelLayers.size() - 1; i >= 0; i--) {
			modelLayers[i]->updateWeights(this->optimizer, this->learningRate);
		}	
	}

// public:

	void setOptimizer(Optimizer<float> *optimizer_) {
		this->optimizer = optimizer_;
	}

	void setLearningRate (float lr_) {
		this->learningRate = lr_;
	}

	float getLearningRate () {
		return this->learningRate;
	}

	void setLoss(LossFunction *loss_) {
		this->loss = loss_;
	}

	Model(Optimizer<float> *optimizer_, float learningRate_, LossFunction *loss_): optimizer(optimizer_), 
		learningRate(learningRate_), loss(loss_){
			modelLayers.clear();
	}

	Model() {
		(void)0;
	}

	void addLayer(Layer *l) {
		modelLayers.push_back(l);
	}

	void trainOne(Tensor<float> input, Tensor<float> targets) {
		float lossValue;
		Tensor<float> outGrads;
		Tensor<float> output = forwardPass(input, false);
		this->loss->operator()(output, targets, lossValue, outGrads);
		calculateModelGradients(outGrads);
		updateModelParameters();
	}

	int correct(Tensor<float> input, Tensor<float> targets, bool shouldPrint) {
		Tensor<float> output = forwardPass(input, false);
		if (shouldPrint) {
			cout << "OUTPUT: \n";
			output.printTensor();
			cout << "TARGET: \n";
			targets.printTensor();
		}
		int mxi = -1;
		float mxv = -199999999.0;
		for (int i=0; i < output.tsize.x; i++) {
			if (output(i,0,0) > mxv) {
				mxi = i;
				mxv = output(i,0,0);
			}
		}

		if (targets(mxi, 0, 0) == 1.0) return 1;
		return 0;
	}

};

#endif