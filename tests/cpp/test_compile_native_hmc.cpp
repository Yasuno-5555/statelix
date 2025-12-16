#include "bayes/hmc.h"
#include <iostream>

int main() {
    statelix::HMCConfig config;
    statelix::HMC hmc(config);
    std::cout << "HMC compiled" << std::endl;
    return 0;
}
