#!/bin/sh
apt-get update >/dev/null
apt-get install -y g++ >/dev/null
g++ -std=c++17 -I/statelix/src -I/statelix/vendor/eigen -c tests/cpp/test_compile_native_hmc.cpp -o hmc.o
