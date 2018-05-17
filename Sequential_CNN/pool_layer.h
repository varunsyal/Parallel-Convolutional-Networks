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

	PoolLayer(int kernelSizeX_, int kernelSizeY_, TensorSize inSize_ ): kernelSizeX(kernelSizeX_), 
		kernelSizeY(kernelSizeY_), inSize(inSize_) {
			assert( (inSize.x % kernelSizeX) == 0 && (inSize.y % kernelSizeY) == 0); 
			in = new Tensor<float> (inSize_.x, inSize_.y, inSize_.z);
			out = new Tensor<float> (inSize.x/kernelSizeX, inSize.y/kernelSizeY, inSize.z);
			outSize = {inSize.x/kernelSizeX, inSize.y/kernelSizeY, inSize.z};
			maxIndices = new Tensor<pair<int,int> > (inSize.x/kernelSizeX, inSize.y/kernelSizeY, inSize.z);
			gradIn = new Tensor<float> (inSize);

	}


	Tensor<float>& operator() (Tensor<float> input) {
		*in = input;
		this->forward();
		return *out;
	}

	//Forward Convolution Pass
	void forward() {
		int mxi = -1, mxj = -1, mxv = -19999999;
		for (int z = 0; z < inSize.z; z++) {
			for (int x = 0; x < outSize.x; x++) {
				for (int y = 0; y < outSize.y; y++) {
					
					for (int i = x*kernelSizeX; i < (x+1)*kernelSizeX; i++) {
						for (int j = y*kernelSizeY; j < (y+1)*kernelSizeY; j++) {
							if (in->get(i,j,z) > mxv) {
								mxv = in->get(i,j,z);
								mxi = i;
								mxj = j;
							}
						}
					}
					out->get(x,y,z) = mxv;
					maxIndices->get(x,y,z) = pair<int,int>(mxi, mxj);
					mxi = -1;
					mxj = -1;
					mxv = -19999999;
				}
			}
		}
	}


	//Calculating gradients for Backpropogation
	void calculateGradients(Tensor<float> gradOut) {
		assert(gradOut.tsize.x == out->tsize.x);
		assert(gradOut.tsize.y == out->tsize.y);
		assert(gradOut.tsize.z == out->tsize.z);
		for (int z = 0; z < inSize.z; z++) {
			for (int x = 0; x < inSize.x; x++) {
				for (int y = 0; y < inSize.y; y++) {
					gradIn->get(x,y,z) = 0.0;
				}
			}
		}

		for (int z = 0; z < inSize.z; z++) {
			for (int x = 0; x < outSize.x; x++) {
				for (int y = 0; y < outSize.y; y++) {
					gradIn->get(maxIndices->get(x,y,z).first, maxIndices->get(x,y,z).second, z) = gradOut(x,y,z);
				}
			}
		}
	}


	void updateWeights(Optimizer<float> optimizer, float learningRate) {
		(void)0;
	}


};

#endif