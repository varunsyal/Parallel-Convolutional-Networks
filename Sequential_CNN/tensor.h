#include<iostream>
#include<assert.h>

using namespace std;

struct TensorSize {
public:
	TensorSize(int x_, int y_, int z_): x(x_), y(y_), z(z_) {}
	TensorSize() {};
	int x, y, z;

	bool operator== (const struct TensorSize& rhs) {
		return (this->x == rhs.x && this->y == rhs.y && this->z == rhs.z);
	}
};

template <class T>
class Tensor {
public:
	T* data;
	struct TensorSize tsize;

	Tensor(int x, int y, int z): tsize({x,y,z}) {
		data = new T[x*y*z];
	}

	Tensor( struct TensorSize tsize_ ): tsize(tsize_) {
		data = new T[tsize_.x * tsize_.y * tsize_.z];
	}

	Tensor<T> operator+ (Tensor<T>& ts) {
		assert(this->tsize == ts.tsize);
		Tensor<T> ret(ts.tsize);
		for (int i=0; i < ts.tsize.x * ts.tsize.y * ts.tsize.z; i++) {
			ret.data[i] = this->data[i] + ts.data[i];
		}
		return ret;
	}

	Tensor<T> operator- (Tensor<T>& ts) {
		assert(this->tsize == ts.tsize);
		Tensor<T> ret(ts.tsize);
		for (int i=0; i < ts.tsize.x * ts.tsize.y * ts.tsize.z; i++) {
			ret.data[i] = this->data[i] - ts.data[i];
		}
		return ret;
	}


	T& operator() (int x, int y, int z) {
		return get(x, y, z);
	}

	T& get(int x, int y, int z) {
		assert(x < this->tsize.x);
		assert(y < this->tsize.y);
		assert(z < this->tsize.z);

		return data[x * (this->tsize.y * this->tsize.z) + y * (this->tsize.z) + z];
	}

	void printSize() {
		cout << "Tensor of size: (" << this->tsize.x << ", " << this->tsize.y << ", " << this->tsize.z << ")\n";
	}

	void printTensor() {
		for (int k = 0; k < this->tsize.z; k++) {
			cout << "\nz = " << k << ": \n";
			for (int i = 0; i < this->tsize.x; i++) {
				cout << "\n\t";
				for (int j = 0; j < this->tsize.y; j++) {
					cout << this->get(i, j, k) << " ";
				}
			}
		}
		cout << endl;
	}

};



