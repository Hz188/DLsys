# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	g++ -O3 -Wall -shared -undefined dynamic_lookup -std=c++11 -fPIC $$(python -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so