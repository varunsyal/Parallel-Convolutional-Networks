#include<cstdlib>
#include "layer.h"
#include "gradient.h"
#include "adagrad_optimizer.h"
#include <vector>
#include <omp.h>

#ifdef __MPI_IMPLEMENTATION__
#include "mpi.h"
#endif

using namespace std;

class ConvLayer: public Layer {
public:
	int kernelSize, numKernels, stride, padding;
	TensorSize inSize;
	vector<Tensor<float> > kernels;
	vector<Tensor<Gradient<float> > > gradKernels;


	ConvLayer(int indx_, int kernelSize_, int numKernels_, int stride_, int padding_, TensorSize inSize_ ): kernelSize(kernelSize_), 
		numKernels(numKernels_), stride(stride_), padding(padding_), inSize(inSize_) {
			this->indx = indx_;
			assert( (inSize.x + 2*padding - kernelSize)%stride == 0 );
			this->type = conv_layer;
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


	Tensor<float> operator() (Tensor<float> input, vector<Tensor<float> > &layer_ins) {
		Tensor<float> *in_ = new Tensor<float> (inSize.x + padding*2, inSize.y + padding*2, inSize.z);
		if (padding == 0)
			*in_ = input;
		else {
			for (int k = 0; k < input.tsize.z; k++) {
				for (int i = 0; i < padding; i++) {
					for (int j=0; j < in_->tsize.y; j++) {
						in_->get(i,j,k) = 0.0;
					}
				}
				for (int i = in_->tsize.x - padding; i < in_->tsize.x; i++) {
					for (int j=0; j < in->tsize.y; j++) {
						in_->get(i,j,k) = 0.0;
					}
				}
				for (int j = 0; j < padding; j++) {
					for (int i=0; i < in_->tsize.x; i++) {
						in_->get(i,j,k) = 0.0;
					}
				}
				for (int j = in_->tsize.y - padding; j < in_->tsize.y; j++) {
					for (int i=0; i < in->tsize.x; i++) {
						in_->get(i,j,k) = 0.0;
					}
				}
				for (int i = padding; i < padding + input.tsize.x; i++) {
					for (int j = padding; j < padding + input.tsize.y; j++) {				
						in_->get(i,j,k) = input(i-padding, j-padding, k);
					}
				}
			}

		}
		layer_ins.push_back(*in_);
		return this->forward(in_);
		
	}

	//Forward Convolution Pass
	Tensor<float> forward(Tensor<float> *in_) {
		Tensor<float> *out_ = new Tensor<float> ((inSize.x + 2*padding - kernelSize)/stride + 1, 
				(inSize.y + padding*2 - kernelSize)/stride + 1, 
				numKernels);
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
								sum += in_->get(x+i, y+j, z) * this->kernels[k](i,j,z);
							}
						}
					}
					out_->get(outx, outy, k) = sum;
					outy++;
				}
				outy = 0;
				outx++;
			}
			outx = 0;
		}
		return *out_;
	}


	//Calculating gradients for Backpropogation
	Tensor<float> calculateGradients(Tensor<float> gradOut, vector<Tensor<float> > layer_ins) {
		assert(gradOut.tsize.x == out->tsize.x);
		assert(gradOut.tsize.y == out->tsize.y);
		assert(gradOut.tsize.z == out->tsize.z);

		Tensor<float> *gradIn_ = new Tensor<float> (inSize);		
		vector<Tensor<Gradient<float> > > gradKernels_;
		for (int i = 0; i < this->numKernels; i++) {
			Tensor<Gradient<float> > gradKernel(kernelSize, kernelSize, inSize.z);
			gradKernels_.push_back(gradKernel);
		}

		for (int x = 0; x < inSize.x; x++) {
			for (int y = 0; y < inSize.y; y++) {
				for (int z = 0; z < inSize.z; z++ ) {
					gradIn_->get(x,y,z) = 0.0;
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
										gradIn_->get(x + i - padding, y + j - padding, z) += 
											gradOut(outx, outy, k) * this->kernels[k](i,j,z);
									}
								}
								gradKernels_[k](i,j,z).value += gradOut(outx, outy, k) * layer_ins[this->indx].get(x+i, y+j, z);
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
		
		#pragma omp critical
		{
		for (int i = 0; i < this->numKernels; i++) {
			
			gradKernels[i] = gradKernels[i] + gradKernels_[i];
		}
		}
		return *gradIn_;
	}


	void updateWeights(Optimizer<float> *optimizer, float learningRate, int batchSize) {
		for (int k = 0; k < numKernels; k++) {
			for (int x = 0; x < kernelSize; x++) {
				for (int y = 0; y < kernelSize; y++) {
					for (int z = 0; z < inSize.z; z++) {
						optimizer->updateWeight(kernels[k].get(x,y,z), gradKernels[k].get(x,y,z), learningRate, batchSize);
						optimizer->updateGradient(gradKernels[k].get(x,y,z));
					}
				}
			}
		}
	}

	void clearGradients() {
		for (int k=0; k < numKernels; k++) {
			for (int x=0; x < kernelSize; x++) {
				for (int y=0; y < kernelSize; y++) {
					for (int z=0; z < inSize.z; z++) {
						gradKernels[k](x,y,z).value = 0.0;	
					}
				}
			}
		}
	}

	void printWeights() {
		for (int i=0; i < kernels.size(); i++) {
			cout << "KERNEL " << i << " :\n";
			kernels[i].printTensor();
		}
	}

#ifdef __MPI_IMPLEMENTATION__

	void serializeGradients(vector<Tensor<Gradient<float> > > &gradKernels_, float *buffer_) {
		int indx_ = 0;
		for (int k=0; k < numKernels; k++) {
			for (int x=0; x < kernelSize; x++) {
				for (int y=0; y < kernelSize; y++) {
					for (int z=0; z < inSize.z; z++) {
						buffer_[indx_++] = gradKernels_[k](x,y,z).value;	
					}
				}
			}
		}
	}

	void deserializeGradients(float *buffer_, vector<Tensor<Gradient<float> > > &gradKernels_ ) {
		int indx_ = 0;
		for (int k=0; k < numKernels; k++) {
			for (int x=0; x < kernelSize; x++) {
				for (int y=0; y < kernelSize; y++) {
					for (int z=0; z < inSize.z; z++) {
						gradKernels_[k](x,y,z).value = buffer_[indx_++];	
					}
				}
			}
		}
	}

	void addGlobalGradients() {
		float* local_buffer = new float[kernelSize * kernelSize * numKernels * inSize.z];
		float* summed_buffer = new float[kernelSize * kernelSize * numKernels * inSize.z];
			
		serializeGradients(gradKernels, local_buffer);

		MPI_Allreduce(local_buffer,
			summed_buffer,
			kernelSize * kernelSize * numKernels * inSize.z,
			MPI_FLOAT,
			MPI_SUM,
			MPI_COMM_WORLD);

		deserializeGradients(summed_buffer, gradKernels);
		delete[] local_buffer;
		delete[] summed_buffer;
	}

#else

	void addGlobalGradients() {
		(void)0;
	}

#endif

};