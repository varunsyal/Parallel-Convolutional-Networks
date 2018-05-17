#ifndef __TENSOR_H_INCLUDED__   
#define __TENSOR_H_INCLUDED__   

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

	Tensor(): tsize({0,0,0}) {
		data = new T[0];
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
			for (int j = 0; j < this->tsize.y; j++) {
				cout << "\n\t";
				for (int i = 0; i < this->tsize.x; i++) {
					cout << this->get(i, j, k) << " ";
				}
			}
		}
		cout << endl;
	}

	Tensor<T> flatten() {
		Tensor<T> res(this->tsize.z * this->tsize.y * this->tsize.x, 1, 1);

		for (int k = 0; k < this->tsize.z; k++) {
			for (int j = 0; j < this->tsize.y; j++) {
				for (int i = 0; i < this->tsize.x; i++) {
					int indx = k * this->tsize.y * this->tsize.x + j * this->tsize.x + i;
					res(indx, 0, 0) = this->get(i, j, k);
				}
			}
		}

		return res;
	}

	Tensor<T> reshape(int x, int y, int z) {
		assert(this->tsize.x * this->tsize.y * this->tsize.z == x * y * z);
		Tensor<T> res(x, y, z);
		Tensor<T> temp;

		if (this->tsize.y != 1 || this->tsize.z != 1) {
			temp = this->flatten();
		} else {
			temp = *this;
		}

		for (int k = 0; k < z; k++) {
			for (int j = 0; j < y; j++) {
				for (int i = 0; i < x; i++) {
					int indx = k * y * x + j * x + i;
					res(i, j, k) = temp(indx, 0, 0);
				}
			}
		}
		return res;
	}

};

#endif