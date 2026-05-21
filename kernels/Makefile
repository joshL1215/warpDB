NVCC = nvcc
INCLUDES = -I.

test_l2norm: tests/test_l2norm.cu l2norm.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test_dotprod: tests/test_dotprod.cu dotprod.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test: test_l2norm test_dotprod
	./test_l2norm
	./test_dotprod

bench: bench.cu l2norm.cu dotprod.cu
	$(NVCC) $(INCLUDES) -O3 $^ -o $@

clean:
	rm -f test_l2norm test_dotprod bench
