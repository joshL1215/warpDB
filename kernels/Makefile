NVCC = nvcc
INCLUDES = -I.

test_l2norm: tests/test_l2norm.cu l2norm.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test_cosine_search: tests/test_cosine_search.cu cosine_search.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test: test_l2norm test_cosine_search
	./test_l2norm
	./test_cosine_search

bench: bench.cu l2norm.cu cosine_search.cu
	$(NVCC) $(INCLUDES) -O3 $^ -o $@

clean:
	rm -f test_l2norm test_cosine_search bench
