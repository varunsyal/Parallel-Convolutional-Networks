#include<iostream>
#include "conv_layer.h"
#include "relu_layer.h"
#include "fc_layer.h"
#include "pool_layer.h"


void testRelu() {
	Tensor<float> t10(5,1,1);
	t10(0,0,0) = 3;
	t10(1,0,0) = -7;
	t10(2,0,0) = 4;
	t10(3,0,0) = -1;
	t10(4,0,0) = 0;

	Tensor<float> outgrads(5,1,1);
	outgrads(0,0,0) = 1;
	outgrads(1,0,0) = 1;
	outgrads(2,0,0) = 1;
	outgrads(3,0,0) = 1;
	outgrads(4,0,0) = 1;

	ReLULayer relu1 = ReLULayer({5,1,1});
	Tensor<float> output = relu1(t10);

	output.printSize();
	output.printTensor();

	cout << "------------------";

	relu1.calculateGradients(outgrads);
	relu1.gradIn->printSize();
	relu1.gradIn->printTensor();

	/*
	OUTPUT:
	Tensor of size: (5, 1, 1)

		z = 0: 

			3 
			0 
			4 
			0 
			0 
		------------------Tensor of size: (5, 1, 1)

		z = 0: 

			1 
			0 
			1 
			0 
			0 
	*/
}


void testFC() {
	Tensor<float> t1(3,1,1);
	t1(0,0,0) = 6;
	t1(1,0,0) = 4;
	t1(2,0,0) = 3;

	Tensor<float> outgrads(2,1,1);
	outgrads(0,0,0) = 1;
	outgrads(1,0,0) = 1;

	FCLayer fc1 = FCLayer(3, 2);

	fc1.weights->get(0,0,0) = 2;
	fc1.weights->get(0,1,0) = 1;
	fc1.weights->get(1,0,0) = 2;
	fc1.weights->get(1,1,0) = 0;
	fc1.weights->get(2,0,0) = -1;
	fc1.weights->get(2,1,0) = 5;

	Tensor<float> output = fc1(t1);

	output.printSize();
	output.printTensor();

	cout << "------------------";

	fc1.calculateGradients(outgrads);
	fc1.gradIn->printSize();
	fc1.gradIn->printTensor();

	/*
	OUTPUT:

		Tensor of size: (2, 1, 1)

	z = 0: 

		17 
		21 
	------------------Tensor of size: (3, 1, 1)

	z = 0: 

		3 
		2 
		4 
	*/

}

void testPool() {

	Tensor<float> t1(4,4,2);
	t1(0,0,0) = 4;
	t1(1,0,0) = 2;
	t1(2,0,0) = 3;
	t1(3,0,0) = 1;
	t1(0,1,0) = 6;
	t1(1,1,0) = 7;
	t1(2,1,0) = -2;
	t1(3,1,0) = 3;
	t1(0,2,0) = 0;
	t1(1,2,0) = 1;
	t1(2,2,0) = 2;
	t1(3,2,0) = 4;
	t1(0,3,0) = 1;
	t1(1,3,0) = 20;
	t1(2,3,0) = 5;
	t1(3,3,0) = 2;

	t1(0,0,1) = 6;
	t1(1,0,1) = 5;
	t1(2,0,1) = 4;
	t1(3,0,1) = 0;
	t1(0,1,1) = -3;
	t1(1,1,1) = -2;
	t1(2,1,1) = -1;
	t1(3,1,1) = 5;
	t1(0,2,1) = 4;
	t1(1,2,1) = 0;
	t1(2,2,1) = 0;
	t1(3,2,1) = 0;
	t1(0,3,1) = -1;
	t1(1,3,1) = 2;
	t1(2,3,1) = 1;
	t1(3,3,1) = -1;
	t1.printTensor();
	cout << "=============";
	Tensor<float> outgrads(2,2,2);
	outgrads(0,0,0) = 1;
	outgrads(1,0,0) = 1;
	outgrads(0,1,0) = 1;
	outgrads(1,1,0) = 1;
	outgrads(0,0,1) = 1;
	outgrads(1,0,1) = 1;
	outgrads(0,1,1) = 1;
	outgrads(1,1,1) = 1;


	PoolLayer pool1 = PoolLayer(2,2,{4,4,2});

	Tensor<float> output = pool1(t1);

	output.printSize();
	output.printTensor();

	cout << "------------------";

	pool1.calculateGradients(outgrads);
	pool1.gradIn->printSize();
	pool1.gradIn->printTensor();

}

void testReshapeAndFlatten() {
	Tensor<float> t1(24, 1, 1);
	for (int i=0; i < 24; i++) {
		t1(i, 0, 0) = i;
	}
	Tensor<float> output = t1.reshape(4,3,2);

	output.printSize();
	output.printTensor();

	cout << "------------------";

	output = output.flatten();

	output.printSize();
	output.printTensor();

	cout << "-----------------";

	output = output.reshape(4,3,2);

	output.printSize();
	output.printTensor();	

	/*
	z = 0: 

	0 1 2 3 
	4 5 6 7 
	8 9 10 11 
	
	z = 1: 

		12 13 14 15 
		16 17 18 19 
		20 21 22 23 
	------------------Tensor of size: (24, 1, 1)

	z = 0: 

		0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 
	-----------------Tensor of size: (4, 3, 2)

	z = 0: 

		0 1 2 3 
		4 5 6 7 
		8 9 10 11 
	z = 1: 

		12 13 14 15 
		16 17 18 19 
		20 21 22 23 
	*/

} 

void testConvolutionLayer() {
	Tensor<float> t1(3,3,1);
	t1(0,0,0) = 3;
	t1(1,0,0) = 4;
	t1(2,0,0) = 1;
	t1(0,1,0) = 5;
	t1(1,1,0) = 6;
	t1(2,1,0) = -2;
	t1(0,2,0) = 0;
	t1(1,2,0) = 9;
	t1(2,2,0) = 1;

	ConvLayer conv1 = ConvLayer(2, 1, 1, 0, {3,3,1} );

	conv1.kernels[0].get(0,0,0) = 2;
	conv1.kernels[0].get(1,0,0) = -1;
	conv1.kernels[0].get(0,1,0) = -2;
	conv1.kernels[0].get(1,1,0) = 1;

	Tensor<float> output = conv1(t1);
	output.printSize();
	output.printTensor();

	cout << "----------------\ngradKernels: \n";

	Tensor<float> outgrads(2,2,1);
	outgrads(0,0,0) = 1;
	outgrads(1,0,0) = 1;
	outgrads(0,1,0) = 1;
	outgrads(1,1,0) = 1;

	conv1.calculateGradients(outgrads);
	cout << "Num of gradKernels = " << conv1.gradKernels.size();
	conv1.gradKernels[0].printSize();
	conv1.gradKernels[0].printTensor();

	cout << "-------------------\ngradIns: \n";

	conv1.gradIn[0].printSize();
	conv1.gradIn[0].printTensor();

	/*
	OUTPUT:

	Tensor of size: (2, 2, 1)

	z = 0: 

		-2 -7 
		13 -3 
	----------------
	gradKernels: 
	Num of gradKernels = 1Tensor of size: (2, 2, 1)

	z = 0: 

		18 9 
		20 14 
	-------------------
	gradIns: 
	Tensor of size: (3, 3, 1)

	z = 0: 

		2 1 -1 
		0 0 0 
		-2 -1 1
	*/

}


int main () {
	testPool();
	testRelu();
	testFC();
	testReshapeAndFlatten();
	testConvolutionLayer();

	return 0;
}