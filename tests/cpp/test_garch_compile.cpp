#include <iostream>
#include <Eigen/Dense>
#include "src/time_series/garch.h"

int main() {
    try {
        statelix::GARCH model(1, 1);
        model.type = statelix::GARCHType::GARCH;
        
        Eigen::VectorXd returns(100);
        returns.setRandom();
        
        std::cout << "Compiling GARCH..." << std::endl;
        // statelix::GARCHResult res = model.fit(returns); // Don't run, just compile check
        std::cout << "GARCH compiled successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
