#include "Sequential_CNN/model_mpi.h"
#include "Sequential_CNN/sgd_optimizer.h"
#include "Sequential_CNN/softmax_cross_entropy.h"

#define __MPI_IMPLEMENTATION__

#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include<time.h>
#include<algorithm>
#include "mpi.h"

#define MASTER 0

using namespace std;

class LeNet: public Model {
public:
	LeNet(Optimizer<float> *optimizer_, float lr, LossFunction* loss_): Model(optimizer_, lr, loss_) {

		Layer *pool0 = new PoolLayer(0, 2, 2, {28, 28, 1});
		this->addLayer(pool0);
		
		Layer *conv1 = new ConvLayer(1, 5, 6, 1, 0, {14, 14, 1});
		this->addLayer(conv1);

		Layer *sigm2 = new SigmoidLayer(2, {10, 10, 6});
		this->addLayer(sigm2);

		Layer *conv3 = new ConvLayer(3, 5, 16, 1, 0, {10, 10, 6});
		this->addLayer(conv3);

		Layer *sigm4 = new SigmoidLayer(4, {6, 6, 16});
		this->addLayer(sigm4);		

		Layer *fc5 = new FCLayer(5, 6*6*16, 10);
		this->addLayer(fc5);

		Layer *sigm6 = new SigmoidLayer(6, {10, 1, 1});
		this->addLayer(sigm6);

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

void shuffleData(vector<Tensor<float> > &data, vector<Tensor<float> > &labels) {
	int sz = data.size();
	vector<pair<Tensor<float>, Tensor<float> > > zippedList;

	for (int i=0; i < sz; i++) {
		zippedList.push_back(pair<Tensor<float>, Tensor<float> >(data[i], labels[i]));
	}
	
	random_shuffle ( zippedList.begin(), zippedList.end() );
	data.clear();
	labels.clear();
	for (int i=0; i < sz; i++) {
		data.push_back(zippedList[i].first);
		labels.push_back(zippedList[i].second);
	}
	zippedList.clear();
}

#define NUM_EPOCHS 500
#define NUM_TRAIN_SAMPLES 500
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define LEARNING_RATE 0.01
#define MINIBATCH 20
#define SGD 0

vector<Tensor<float> > trainData;
vector<Tensor<float> > trainLabels;

int main(int argc, char *argv[])
{
	int numtasks, taskid;
		/***** Initializations *****/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

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

	int corr;
	float accuracy;
	double start, finish;
	double elapsed;
	start = omp_get_wtime();
	
	if (taskid == MASTER) {
		printf("Number of MPI tasks: %d \n", numtasks);
	}

	//-------------------Train Network ----------------------
	
	
	for (int e = 0, c = 1; e < NUM_EPOCHS; e++) {
		shuffleData(trainData, trainLabels);

		// // DECAYING LERANING RATE
		// if (e > 0 && ((e%(8*c))==0)) {
		//	 Lenet.setLearningRate(Lenet.getLearningRate()/3.0);
		//   c *= 2;
		// }
		
	  	if (taskid == MASTER) {
	  		corr = 0;
		  	for (int i = 0; i < NUM_TRAIN_SAMPLES; i++) {
		  		corr += Lenet.correct(trainData[i], trainLabels[i], false);
		  	}  	
		  	accuracy = (float) corr / (float) NUM_TRAIN_SAMPLES;
		  	cout << "Accuracy = " << accuracy << endl;
			
			cout << "----------------------EPOCH: " << e << "------------------------" << endl;
		}

		// shuffle(trainData, trainLabels);
		vector<Tensor<float> >::iterator itData1, itData2, itLabel1, itLabel2;
		itData1 = trainData.begin();
		itLabel1 = trainLabels.begin();
		int kk = 0;
		while (itData1 < trainData.end()) {
			itData2 = itData1 + MINIBATCH;
			itLabel2 = itLabel1 + MINIBATCH;
			if (itData2 >= trainData.end()) itData2 = trainData.end();
			if (itLabel2 >= trainLabels.end()) itLabel2 = trainLabels.end();
			vector<Tensor<float> > trainDataBatch(itData1, itData2);
			vector<Tensor<float> > trainLabelBatch(itLabel1, itLabel2);
			Lenet.trainBatch(trainDataBatch, trainLabelBatch, true);
			MPI_Barrier(MPI_COMM_WORLD);
			itData1 = itData2;
			itLabel1 = itLabel2;
		}

	  	
	}

	
  	if (taskid == MASTER) {
  		corr = 0;
	  	for (int i = 0; i < NUM_TRAIN_SAMPLES; i++) {
	  		corr += Lenet.correct(trainData[i], trainLabels[i], false);
	  	}  	
  		accuracy = (float) corr / (float) NUM_TRAIN_SAMPLES;
  		cout << "Accuracy = " << accuracy << endl;
  	}

	finish = omp_get_wtime();
	elapsed = (finish - start);
	cout << "Time: " << elapsed << endl;
  	return 0;
}
