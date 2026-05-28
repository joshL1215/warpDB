#!/bin/bash

nvcc -std=c++17 vector_engine.cpp ../kernels/cosine_search.cu -I../kernels -o testcompile
