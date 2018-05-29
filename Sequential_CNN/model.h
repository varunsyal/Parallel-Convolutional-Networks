#ifndef __MODEL_H_INCLUDED__   
#define __MODEL_H_INCLUDED__  

#include "conv_layer.h"
#include "pool_layer.h"
#include "fc_layer.h"
#include "relu_layer.h"
#include "sigmoid_layer.h"
#include "softmax_cross_entropy.h"
#include <vector>
#include <omp.h>

using namespace std;

class Model {
public:
	int x = 0;
	float learningRate;
	Optimizer<float> *optimizer;
	vector<Layer*> modelLayers;
	LossFunction *loss;

	Tensor<float> forwardPass(Tensor<float> input, vector<Tensor<float> >& layer_ins, bool prnt) {
		Tensor<float> x = input;
		for (int i = 0; i < modelLayers.size(); i++) {
			if (modelLayers[i]->type == fc_layer) {
				x = x.flatten();
			}
			Tensor<float> tmp = modelLayers[i]->operator()(x, layer_ins);
			x = tmp;
			if (prnt) {
				cout << "=======LAYER " << i << modelLayers[i]->type << "=============\n\n";
				x.printTensor();
				if (i==1) {
					modelLayers[1] -> printWeights();
				}	
			}
		}


		return x;
	}

	void calculateModelGradients(Tensor<float> outGrads, vector<Tensor<float> > layer_ins) {
		Tensor<float> x = outGrads;
		for (int i = modelLayers.size() - 1; i >= 0; i--) {
			if (i+1 < modelLayers.size() && modelLayers[i+1] -> type == fc_layer) {
				x = x.reshape(modelLayers[i]->out->tsize);
			}
			x = modelLayers[i]->calculateGradients(x, layer_ins);
		}
	}

	void updateModelParameters(int batchSize) {
		for (int i = modelLayers.size() - 1; i >= 0; i--) {
			modelLayers[i]->updateWeights(this->optimizer, this->learningRate, batchSize);
		}	
	}

	void clearModelGradients() {
		for (int i = 0; i < modelLayers.size(); i++) {
			modelLayers[i] -> clearGradients();
		}
	}

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

	void trainOne(Tensor<float> input, Tensor<float> targets, bool b) {
		float lossValue;
		Tensor<float> outGrads;
		vector<Tensor<float> > layer_ins;
		Tensor<float> output = forwardPass(input, layer_ins, false);
		this->loss->operator()(output, targets, lossValue, outGrads);
		
		if (b == true) {
			output.printTensor();
		}
		clearModelGradients();
		calculateModelGradients(outGrads, layer_ins);
		updateModelParameters(1);
	}

	void trainBatch(vector<Tensor<float> > inputs, vector<Tensor<float> > targets, bool b) {
		int tid;
		
		#pragma omp parallel private(tid)
		{
		tid = omp_get_thread_num();

		clearModelGradients();
		
		#pragma omp barrier

		#pragma omp for
		for (int i=0; i < inputs.size(); i++) {
			float lossValue;
			vector<Tensor<float> > layer_ins;
			Tensor<float> output;

			if (i==0 ) {
				output = forwardPass(inputs[i], layer_ins, false);	
			} else {
				output = forwardPass(inputs[i], layer_ins, false);
			}

			Tensor<float> outg;
			this->loss->operator()(output, targets[i], lossValue, outg);

			calculateModelGradients(outg, layer_ins);
		}

		

		// if (b == true) {
		// 	 outGrads[0].printTensor();
		// }
		
		#pragma omp barrier		

		updateModelParameters(inputs.size());

		}	


	}

	int correct(Tensor<float> input, Tensor<float> targets, bool shouldPrint) {
		vector<Tensor<float> > layer_ins;
		Tensor<float> output = forwardPass(input, layer_ins, false);
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