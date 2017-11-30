#include <iostream>
#include "SGD.h"
#include "LR.h"

using namespace std;

int main() {
    LR lr;
    lr.load_train_data("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte");
    lr.load_test_data("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte");
    lr.train();
    lr.test();
    return 0;
}
