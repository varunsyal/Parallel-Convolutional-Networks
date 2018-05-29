#include "activation_layer.h"

using namespace std;

class ReLULayer: public ActivationLayer {
public:
	ReLULayer(int indx_, TensorSize inSize_ ): ActivationLayer(indx_, inSize_) {
		this->type = relu_layer;
	}

	float _activate_(float in) {
		if (in > 0) return in;
		else return 0.0;
	}

	float _gradient_(float in) {
		if (in > 0) return 1.0;
		else return 0.0;
	}

};