#include<cstdlib>
#include "layer.h"
#include "gradient.h"
#include "adagrad_optimizer.h"
#include<vector>

using namespace std;

class FCLayer: public Layer {
public:
	int inSize, outSize;
	Tensor<float> *weights;
	Tensor<Gradient<float> > *weightGrads;

	FCLayer(int inSize_, int outSize_): inSize(inSize_), outSize(outSize_) {
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

	Tensor<float>& operator() (Tensor<float> input) {
		assert(input.tsize.y == 1);
		assert(input.tsize.z == 1);
		
		*in = input;
		this->forward();
		return *out;
	}

	//Forward Convolution Pass
	void forward() {
		for (int i = 0; i < outSize; i++) {
			out->get(i,0,0) = 0.0;
			for (int j = 0; j < inSize; j++) {
				out->get(i,0,0) += in->get(j,0,0) * weights->get(j,i,0);
			}
		}
	}

	void calculateGradients(Tensor<float> gradOut) {
		assert(gradOut.tsize.x == outSize && gradOut.tsize.y == 1 && gradOut.tsize.z == 1);
		for (int i = 0; i < inSize; i++) {
			gradIn->get(i,0,0) = 0.0;
		}

		for (int i = 0; i < inSize; i++) {
			for (int j = 0; j < outSize; j++) {
				weightGrads->get(i,j,0).value = 0.0;
			}
		}
		
		for (int i = 0; i < outSize; i++) {
			out->get(i,0,0) = 0.0;
			for (int j = 0; j < inSize; j++) {
				gradIn->get(j,0,0) += gradOut(i,0,0) * weights->get(j,i,0);
				weightGrads->get(j,i,0).value += gradOut(i,0,0) * in->get(j,0,0);
			}
		}
	}

	void updateWeights(Optimizer<float> *optimizer, float learningRate) {
		for (int i = 0; i < inSize; i++) {
			for (int j = 0; j < outSize; j++) {
				optimizer->updateWeight(weights->get(i,j,0), weightGrads->get(i,j,0), learningRate);
				optimizer->updateGradient(weightGrads->get(i,j,0));
			}
		}
	}
};