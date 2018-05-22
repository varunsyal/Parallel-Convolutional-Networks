#include "Sequential_CNN/model.h"
#include "Sequential_CNN/sgd_optimizer.h"
#include "Sequential_CNN/softmax_cross_entropy.h"

#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

class LeNet: public Model {
public:
	LeNet(Optimizer<float> *optimizer_, float lr, LossFunction* loss_): Model(optimizer_, lr, loss_) {

		// Layer *conv1 = new ConvLayer(5, 6, 1, 2, {28, 28, 1});
		// this->addLayer(conv1);

		// Layer *relu1 = new SigmoidLayer({28, 28, 6});
		// this->addLayer(relu1);

		Layer *pool2 = new PoolLayer(2, 2, {28, 28, 1});
		this->addLayer(pool2);

		Layer *conv3 = new ConvLayer(5, 16, 1, 0, {14, 14, 1});
		this->addLayer(conv3);

		Layer *sigm3 = new SigmoidLayer({10, 10, 16});
		this->addLayer(sigm3);

		Layer *pool4 = new PoolLayer(2, 2, {10, 10, 16});
		this->addLayer(pool4);

		Layer *conv5 = new ConvLayer(5, 120, 1, 0, {5, 5, 16});
		this->addLayer(conv5);

		Layer *sigm5 = new SigmoidLayer({1,1,120});
		this->addLayer(sigm5);

		Layer *fc6 = new FCLayer(120, 84);
		this->addLayer(fc6);

		Layer *sigm6 = new SigmoidLayer({84, 1, 1});
		this->addLayer(sigm6);

		Layer *fc7 = new FCLayer(84, 10);
		this->addLayer(fc7);

	}
};

int reverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void readMNIST(string filename, int NumberOfImages, int x_max, int y_max, vector<Tensor<float> > &arr)
{
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= min(reverseInt(number_of_images), NumberOfImages);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
        	arr.push_back(Tensor<float>(x_max, y_max, 1));
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i](r, c, 0)= (float)temp;
                }
            }
        }
    }
}

void readMnistLabel(string filename, int NumberOfImages, int NumberOfLabels, vector<Tensor<float> > &vec)
{
    ifstream file (filename, ios::binary);
    // vec.resize(NumberOfImages, Tensor<float>(NumberOfLabels, 1, 1));
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = min(reverseInt(number_of_images), NumberOfImages);
        for(int i = 0; i < number_of_images; i++)
        {
        	vec.push_back(Tensor<float>(NumberOfLabels, 1, 1));
            unsigned char temp = 0;
            for (int j = 0; j < NumberOfLabels; j++) {
            	vec[i](j, 0, 0) = 0.0;	
            }
            file.read((char*) &temp, sizeof(temp));
            vec[i].get((int)temp, 0, 0) = 1.0;
        }
    }

}

void normalizeData(vector<Tensor<float> > &vec) {
	Tensor<float> mean(vec[0].tsize);
	Tensor<float> stdv(vec[0].tsize);
	clear(mean);
	clear(stdv);

	for (int i=0; i < vec.size(); i++) {
		mean = mean + vec[i];
	}
	mean = mean / float(vec.size());

	for (int i=0; i < vec.size(); i++) {
		vec[i] = vec[i] - mean;
		stdv = vec[i].square() + stdv;
	}
	stdv = stdv + 1.0;
	stdv = stdv.sqrt_();

	for (int i=0; i < vec.size(); i++) {
		vec[i] = vec[i] / stdv;
		// vec[i].printTensor();
	}
}
 
#define NUM_EPOCHS 10
#define NUM_TRAIN_SAMPLES 100
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define LEARNING_RATE 0.1

vector<Tensor<float> > trainData;
vector<Tensor<float> > trainLabels;

int main()
{
	int corr;
	float accuracy;

	// ----------------- Read MNIST Data--------------------
	readMNIST("./Datasets/t10k-images.idx3-ubyte", NUM_TRAIN_SAMPLES, INPUT_WIDTH, INPUT_HEIGHT, trainData); 
	readMnistLabel("./Datasets/t10k-labels.idx1-ubyte", NUM_TRAIN_SAMPLES, 10, trainLabels);

	// ------------------Pre-process Data -------------------
	normalizeData(trainData);

	// ------------------Create Network model ---------------
	Optimizer<float> *optimizer = new SGDOptimizer<float>();
	float lr = LEARNING_RATE;
	LossFunction* loss = new SoftmaxCrossEntropy();
	LeNet Lenet(optimizer, lr, loss);

	//-------------------Train Network ----------------------
	int start_s=clock();
	for (int e = 0; e < NUM_EPOCHS; e++) {
		cout << "----------------------EPOCH: " << e << "------------------------" << endl;
	  	for (int i = 0; i < NUM_TRAIN_SAMPLES; i++) {
	  		Lenet.trainOne(trainData[i], trainLabels[i]);
	  	}

	  	corr = 0;
	  	for (int i = 0; i < NUM_TRAIN_SAMPLES; i++) {
	  		corr += Lenet.correct(trainData[i], trainLabels[i], false);
	  	}  	
	  	accuracy = (float) corr / (float) NUM_TRAIN_SAMPLES;
	  	cout << "Accuracy = " << accuracy << endl;
	}
	int stop_s=clock();
	cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC) << endl;
  	return 0;
}
