#ifndef __POOL_H_INCLUDED__   
#define __POOL_H_INCLUDED__  

#include<cstdlib>
#include "layer.h"
#include "gradient.h"
#include "adagrad_optimizer.h"
#include<vector>
#include<utility>
// #define INF 19999999

using namespace std;

class PoolLayer: public Layer {
public:
	int kernelSizeX, kernelSizeY;
	TensorSize inSize, outSize;
	Tensor<pair<int,int> >* maxIndices;

	PoolLayer(int indx_, int kernelSizeX_, int kernelSizeY_, TensorSize inSize_ ): kernelSizeX(kernelSizeX_), 
		kernelSizeY(kernelSizeY_), inSize(inSize_) {
			this->indx = indx_;
			assert( (inSize.x % kernelSizeX) == 0 && (inSize.y % kernelSizeY) == 0); 
			this->type = pool_layer;
			in = new Tensor<float> (inSize_.x, inSize_.y, inSize_.z);
			out = new Tensor<float> (inSize.x/kernelSizeX, inSize.y/kernelSizeY, inSize.z);
			outSize = {inSize.x/kernelSizeX, inSize.y/kernelSizeY, inSize.z};
			maxIndices = new Tensor<pair<int,int> > (inSize.x/kernelSizeX, inSize.y/kernelSizeY, inSize.z);
			gradIn = new Tensor<float> (inSize);

	}


	Tensor<float> operator() (Tensor<float> input, vector<Tensor<float> > &layer_ins) {
		Tensor<float> *in_ = new Tensor<float> (inSize.x, inSize.y, inSize.z);
		*in_ = input;
		layer_ins.push_back(*in_);
		return this->forward(in_);
	}

	//Forward Convolution Pass
	Tensor<float> forward(Tensor<float> *in_) {
		Tensor<float> *out_ = new Tensor<float> (inSize.x/kernelSizeX, inSize.y/kernelSizeY, inSize.z);
		int mxi = -1, mxj = -1;
		float mxv = -19999999.0;
		for (int z = 0; z < inSize.z; z++) {
			for (int x = 0; x < outSize.x; x++) {
				for (int y = 0; y < outSize.y; y++) {
					
					for (int i = x*kernelSizeX; i < (x+1)*kernelSizeX; i++) {
						for (int j = y*kernelSizeY; j < (y+1)*kernelSizeY; j++) {
							if (in_->get(i,j,z) > mxv) {
								mxv = in_->get(i,j,z);
								mxi = i;
								mxj = j;
							}
						}
					}
					out_->get(x,y,z) = mxv;
					maxIndices->get(x,y,z) = pair<int,int>(mxi, mxj);
					mxi = -1;
					mxj = -1;
					mxv = -19999999;
				}
			}
		}
		return *out_;
	}


	//Calculating gradients for Backpropogation
	Tensor<float> calculateGradients(Tensor<float> gradOut, vector<Tensor<float> > layer_ins) {
		Tensor<float> *gradIn_ = new Tensor<float> (inSize);

		assert(gradOut.tsize.x == out->tsize.x);
		assert(gradOut.tsize.y == out->tsize.y);
		assert(gradOut.tsize.z == out->tsize.z);
		
		for (int z = 0; z < inSize.z; z++) {
			for (int x = 0; x < inSize.x; x++) {
				for (int y = 0; y < inSize.y; y++) {
					gradIn_->get(x,y,z) = 0.0;
				}
			}
		}

		for (int z = 0; z < inSize.z; z++) {
			for (int x = 0; x < outSize.x; x++) {
				for (int y = 0; y < outSize.y; y++) {
					gradIn_->get(maxIndices->get(x,y,z).first, maxIndices->get(x,y,z).second, z) = gradOut(x,y,z);
				}
			}
		}
		return *gradIn_;
		
	}


	void updateWeights(Optimizer<float> *optimizer, float learningRate, int batchSize) {
		(void)0;
	}

	void clearGradients() {
		(void)0;
	}

	void printWeights() {
		(void)0;
	}

	void addGlobalGradients() {
		(void)0;
	}
};

#endif