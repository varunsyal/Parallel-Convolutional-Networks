#include "activation_layer.h"

using namespace std;

class ReLULayer: public ActivationLayer {
public:
	ReLULayer(TensorSize inSize_ ): ActivationLayer(inSize_) {}

	float _activate_(float in) {
		if (in > 0) return in;
		else return 0.0;
	}

	float _gradient_(float in) {
		if (in > 0) return 1.0;
		else return 0.0;
	}

};