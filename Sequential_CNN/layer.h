#include "tensor.h"

using namespace std;

enum Type {conv_layer, pool_layer, relu_layer, fc_layer};

class Layer {
public:
	Tensor<float> *in;
	Tensor<float> *out;
	Tensor<float> *gradIn;

};