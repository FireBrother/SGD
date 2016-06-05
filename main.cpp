#include <iostream>
#include "SGD.h"
#include "LR.h"

using namespace std;

int main() {
    LR lr;
    lr.load_train_data("data/a1a");
    lr.load_test_data("data/a1a.t");
    lr.train();
    lr.test();
    return 0;
}
