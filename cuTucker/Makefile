CXX=g++

LIB_FLAGS = -larmadillo -llapack -lblas -DARMA_DONT_USE_WRAPPER

OPT = -O2 -mcmodel=medium  -fopenmp -w -std=c++11

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: SGD_FastTucker demo


SGD_FastTucker: SGD_FastTucker.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $< $(LIB_FLAGS)

demo: SGD_FastTucker.cpp

	g++ -std=c++11 -o SGD_FastTucker SGD_FastTucker.cpp -O2 -fopenmp -w -mcmodel=medium -larmadillo -llapack -lblas -DARMA_DONT_USE_WRAPPER	
	./SGD_FastTucker ./Data/movielens_tensor.train ./Data/movielens_tensor.test 32 3 32 32 32


	
.PHONY: clean

clean:
	rm -f SGD_FastTucker
