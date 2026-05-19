NVCC = nvcc
INCLUDES = -I.

test_invnorm: tests/test_invnorm.cu invnorm.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test_scalerows: tests/test_scalerows.cu scalerows.cu invnorm.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test_dotprod: tests/test_dotprod.cu dotprod.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test: test_invnorm test_scalerows test_dotprod
	./test_invnorm
	./test_scalerows
	./test_dotprod

bench: bench.cu invnorm.cu scalerows.cu dotprod.cu
	$(NVCC) $(INCLUDES) -O3 $^ -o $@

clean:
	rm -f test_invnorm test_scalerows test_dotprod bench
