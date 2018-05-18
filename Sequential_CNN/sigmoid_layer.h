#include "activation_layer.h"

using namespace std;

class SigmoidLayer: public ActivationLayer {
public:
	SigmoidLayer(TensorSize inSize_ ): ActivationLayer(inSize_) {
		this->type = relu_layer;
	}

	float _activate_(float in) {
		return 2.0 / (1.0 + exp(-in)) - 1.0;
	}

	float _gradient_(float in) {
		return 2.0 * exp(-in) / (1.0 + exp(-in)) / (1.0 + exp(-in));
	}

};