CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -pedantic -I..

TARGET = vector_engine_tests
SOURCES = test_vector_engine.cpp ../vector_engine.cpp

.PHONY: test clean

test: $(TARGET)
	./$(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)
