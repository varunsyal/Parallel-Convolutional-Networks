#ifndef __GRAD_H_INCLUDED__   
#define __GRAD_H_INCLUDED__  

using namespace std;

template <class T>
class Gradient {
public:
	T value;
	T prevValue;

	Gradient<T> () {
		value = prevValue = 0;
	}
	template <class U>
	friend ostream& operator<< (ostream& stream, const Gradient<U>& grad); 
};

template <class U>
ostream& operator<< (ostream& stream, const Gradient<U>& grad) {
		stream << grad.value;
}

#endif