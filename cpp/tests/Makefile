NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -I.. -I../../kernels

TARGET = vector_engine_tests
OBJECTS = vector_engine.o cosine_search.o

.PHONY: test clean

test: $(TARGET)
	./$(TARGET)

$(TARGET): test_vector_engine.cpp $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) test_vector_engine.cpp $(OBJECTS) -o $(TARGET)

vector_engine.o: ../vector_engine.cpp ../vector_engine.h
	$(NVCC) $(NVCCFLAGS) -Dmain=vector_engine_internal_main -c ../vector_engine.cpp -o vector_engine.o

cosine_search.o: ../../kernels/cosine_search.cu ../../kernels/kernels.h
	$(NVCC) $(NVCCFLAGS) -c ../../kernels/cosine_search.cu -o cosine_search.o

clean:
	rm -f $(TARGET) $(OBJECTS)
