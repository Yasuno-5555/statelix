#include "time_series/garch.h"
#include <iostream>

int main() {
    statelix::GARCH model(1, 1);
    std::cout << "GARCH created" << std::endl;
    return 0;
}
