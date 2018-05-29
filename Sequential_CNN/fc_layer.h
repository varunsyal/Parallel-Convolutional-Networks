#include<cstdlib>
#include "layer.h"
#include "gradient.h"
#include "adagrad_optimizer.h"
#include<vector>
#include<omp.h>

using namespace std;

class FCLayer: public Layer {
public:
	int inSize, outSize;
	Tensor<float> *weights;
	Tensor<Gradient<float> > *weightGrads;

	FCLayer(int indx_, int inSize_, int outSize_): inSize(inSize_), outSize(outSize_) {
			this->indx = indx_;
			this->type = fc_layer;
			in = new Tensor<float> (inSize_, 1, 1);
			out = new Tensor<float> (outSize_, 1, 1);
			gradIn = new Tensor<float> (inSize_, 1, 1);
			weights = new Tensor<float> (inSize_, outSize_, 1);
			weightGrads = new Tensor<Gradient<float> > (inSize_, outSize_, 1);

			for (int i = 0; i < inSize_; i++) {
				for (int j=0; j < outSize_; j++) {
					weights->get(i,j,0) = rand() * 1.0 / inSize_ / RAND_MAX;
				}
			}
	}

	Tensor<float> operator() (Tensor<float> input, vector<Tensor<float> > &layer_ins) {
		assert(input.tsize.y == 1);
		assert(input.tsize.z == 1);
		
		Tensor<float> *in_ = new Tensor<float> (inSize, 1, 1);
		*in_ = input; 
		// cout << "<<<<";in_ -> printSize();cout << ">>>>>";
		layer_ins.push_back(*in_);
		return this->forward(in_);
	}

	//Forward Convolution Pass
	Tensor<float> forward(Tensor<float> *in_) {
	// #pragma omp parallel
	// {
		// auto tid = omp_get_thread_num();
	 //  	if (tid == 0){
		//     auto nthreads = omp_get_num_threads();
		//     // printf("Starting matrix multiple example with %d threads\n",nthreads);
		//     // printf("Initializing matrices...\n");
	 //    }
		Tensor<float> *out_ = new Tensor<float> (outSize, 1, 1);
		for (int i = 0; i < outSize; i++) {
			out_->get(i,0,0) = 0.0;
			for (int j = 0; j < inSize; j++) {
				out_->get(i,0,0) += in_->get(j,0,0) * weights->get(j,i,0);
			}
		}

		return *out_;

	// }
	}

	Tensor<float> calculateGradients(Tensor<float> gradOut, vector<Tensor<float> > layer_ins) {
		Tensor<float> *gradIn_ = new Tensor<float> (inSize, 1, 1);
		Tensor<Gradient<float> > *weightGrads_ = new Tensor<Gradient<float> > (inSize, outSize, 1);

		assert(gradOut.tsize.x == outSize && gradOut.tsize.y == 1 && gradOut.tsize.z == 1);
		// cerr << "1";
		for (int i = 0; i < inSize; i++) {
			gradIn_->get(i,0,0) = 0.0;
		}
		// cerr << "2{{  "; layer_ins[this->indx].printSize();
		
		for (int i = 0; i < outSize; i++) {
			for (int j = 0; j < inSize; j++) {
				gradIn_->get(j,0,0) += gradOut(i,0,0) * weights->get(j,i,0);
				weightGrads_->get(j,i,0).value += gradOut(i,0,0) * layer_ins[this->indx].get(j,0,0);
			}
		}
		// cerr << "3";
		#pragma omp critical
		{
		*weightGrads = *weightGrads_ + *weightGrads;
		}
		// cerr << "4";
		return *gradIn_;
	}

	void updateWeights(Optimizer<float> *optimizer, float learningRate, int batchSize) {
		for (int i = 0; i < inSize; i++) {
			for (int j = 0; j < outSize; j++) {
				optimizer->updateWeight(weights->get(i,j,0), weightGrads->get(i,j,0), learningRate, batchSize);
				optimizer->updateGradient(weightGrads->get(i,j,0));
			}
		}
	}

	void clearGradients() {
		for (int i = 0; i < inSize; i++) {
			for (int j = 0; j < outSize; j++) {
				weightGrads->get(i,j,0).value = 0.0;
			}
		}
	}

	void printWeights() {
		weights->printTensor();
	}
};