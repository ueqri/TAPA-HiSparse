PROJ_ROOT = ..

CXX := g++

CXXFLAGS += -Wall -std=c++14
CXXFLAGS += -I$(PROJ_ROOT)/sw/include
LDFLAGS += -lrt -lstdc++ -ltapa -lfrt -lglog -lgflags -lOpenCL

CXXFLAGS += -I$(CNPY_INCLUDE)
LDFLAGS += -L$(CNPY_LIB) -lcnpy

CXXFLAGS += -Wno-maybe-uninitialized
CXXFLAGS += -Wno-uninitialized
CXXFLAGS += -Wno-int-in-bool-context
CXXFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unused-function
CXXFLAGS += -Wno-unused-label
CXXFLAGS += -Wno-sign-compare

.PHONY: host benchmark
host: host.cpp $(PROJ_ROOT)/src/spmv.cpp
	$(CXX) $(CXXFLAGS) -g $^ -o host $(LDFLAGS)

benchmark: benchmark.cpp $(PROJ_ROOT)/src/spmv.cpp
	$(CXX) $(CXXFLAGS) -O3 $^ -o benchmark $(LDFLAGS)

.PHONY: clean cleanall
clean:
	rm -rf emconfig.json
	rm -rf .run *.log *.csv

cleanall: clean
	rm -rf host benchmark
