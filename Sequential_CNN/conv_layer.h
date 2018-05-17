#include<cstdlib>
#include "layer.h"
#include "gradient.h"
#include "adagrad_optimizer.h"
#include<vector>

using namespace std;

class ConvLayer: public Layer {
public:
	int kernelSize, numKernels, stride, padding;
	TensorSize inSize;
	vector<Tensor<float> > kernels;
	vector<Tensor<Gradient<float> > > gradKernels;


	ConvLayer(int kernelSize_, int numKernels_, int stride_, int padding_, TensorSize inSize_ ): kernelSize(kernelSize_), 
		numKernels(numKernels_), stride(stride_), padding(padding_), inSize(inSize_) {
			assert( (inSize.x + 2*padding - kernelSize)%stride == 0 );
			
			in = new Tensor<float> (inSize_.x + padding_*2, inSize_.y + padding_*2, inSize_.z);
			out = new Tensor<float> ((inSize_.x + 2*padding_ - kernelSize_)/stride_ + 1, 
				(inSize_.y + padding_*2 - kernelSize_)/stride_ + 1, 
				numKernels_);
			gradIn = new Tensor<float> (inSize_);


			//Initialize Kernels
			float xavInputSize = kernelSize_ * kernelSize_ * inSize_.z;
			for (int i = 0; i < this->numKernels; i++) {
				Tensor<float> kernel(kernelSize_, kernelSize_, inSize_.z);
				Tensor<Gradient<float> > gradKernel(kernelSize_, kernelSize_, inSize_.z);
				for (int j = 0; j < kernelSize_ * kernelSize_ * inSize_.z; j++) {
					kernel.data[j] =  rand() * 1.0 / xavInputSize / RAND_MAX;
				}
				kernels.push_back(kernel);
				gradKernels.push_back(gradKernel);
			}
	}


	Tensor<float>& operator() (Tensor<float> input) {
		if (padding == 0)
			*in = input;
		else {
			for (int k = 0; k < input.tsize.z; k++) {
				for (int i = 0; i < padding; i++) {
					for (int j=0; j < in->tsize.y; j++) {
						in->get(i,j,k) = 0.0;
					}
				}
				for (int i = in->tsize.x - padding; i < in->tsize.x; i++) {
					for (int j=0; j < in->tsize.y; j++) {
						in->get(i,j,k) = 0.0;
					}
				}
				for (int j = 0; j < padding; j++) {
					for (int i=0; i < in->tsize.x; i++) {
						in->get(i,j,k) = 0.0;
					}
				}
				for (int j = in->tsize.y - padding; j < in->tsize.y; j++) {
					for (int i=0; i < in->tsize.x; i++) {
						in->get(i,j,k) = 0.0;
					}
				}
				for (int i = padding; i < padding + input.tsize.x; i++) {
					for (int j = padding; j < padding + input.tsize.y; j++) {				
						in->get(i,j,k) = input(i-padding, j-padding, k);
					}
				}
			}

		}
		this->forward();
		return *out;
	}

	//Forward Convolution Pass
	void forward() {
		assert( (inSize.x + 2*padding - kernelSize)%stride == 0 );
		int outx = 0, outy = 0;
		for (int k = 0; k < numKernels; k++) {
			for (int x = 0; x < inSize.x + padding*2; x += stride) {
				if (x + kernelSize > inSize.x + padding*2) break;
				for (int y = 0; y < inSize.y + padding*2; y += stride) {
					if (y + kernelSize > inSize.y + padding*2) break;
					float sum = 0.0;
					for (int i = 0; i < kernelSize; i++) {
						for (int j = 0; j < kernelSize; j++) {
							for (int z = 0; z < inSize.z; z++) {
								sum += this->in->get(x+i, y+j, z) * this->kernels[k](i,j,z);
							}
						}
					}
					this->out->get(outx, outy, k) = sum;
					outy++;
				}
				outy = 0;
				outx++;
			}
			outx = 0;
		}
	}


	//Calculating gradients for Backpropogation
	void calculateGradients(Tensor<float> gradOut) {
		assert(gradOut.tsize.x == out->tsize.x);
		assert(gradOut.tsize.y == out->tsize.y);
		assert(gradOut.tsize.z == out->tsize.z);
		for (int x = 0; x < inSize.x; x++) {
			for (int y = 0; y < inSize.y; y++) {
				for (int z = 0; z < inSize.z; z++ ) {
					gradIn->get(x,y,z) = 0.0;
				}
			}
		}
		for (int k=0; k < numKernels; k++) {
			for (int x=0; x < kernelSize; x++) {
				for (int y=0; y < kernelSize; y++) {
					for (int z=0; z < inSize.z; z++) {
						gradKernels[k](x,y,z).value = 0.0;	
					}
				}
			}
		}

		assert( (inSize.x + 2*padding - kernelSize)%stride == 0 );
		int outx = 0, outy = 0;
		for (int k = 0; k < numKernels; k++) {
			for (int x = 0; x < inSize.x + padding*2; x += stride) {
				if (x + kernelSize > inSize.x + padding*2) break;
				for (int y = 0; y < inSize.y + padding*2; y += stride) {
					
					if (y + kernelSize > inSize.y + padding*2) break;
					float sum = 0.0;
					for (int i = 0; i < kernelSize; i++) {
						for (int j = 0; j < kernelSize; j++) {
							for (int z = 0; z < inSize.z; z++) {
								if (x + i - padding >= 0 && x + i - padding < inSize.x) {
									if (y + j - padding >= 0 && y + j - padding < inSize.y) {
										gradIn->get(x + i - padding, y + j - padding, z) += 
											gradOut(outx, outy, k) * this->kernels[k](i,j,z);
									}
								}
								gradKernels[k](i,j,z).value += gradOut(outx, outy, k) * this->in->get(x+i, y+j, z);
							}
						}
					}
					outy++;
				}
				outy=0;
				outx++;
			}
			outx=0;
		}	
	}


	void updateWeights(Optimizer<float> optimizer, float learningRate) {
		for (int k = 0; k < numKernels; k++) {
			for (int x = 0; x < kernelSize; x++) {
				for (int y = 0; y < kernelSize; y++) {
					for (int z = 0; z < inSize.z; z++) {
						optimizer.updateWeight(kernels[k].get(x,y,z), gradKernels[k].get(x,y,z), learningRate);
						optimizer.updateGradient(gradKernels[k].get(x,y,z));
					}
				}
			}
		}
	}


};