NVCC = nvcc
INCLUDES = -I.

test_invnorm: tests/test_invnorm.cu invnorm.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test_scalerows: tests/test_scalerows.cu scalerows.cu invnorm.cu
	$(NVCC) $(INCLUDES) $^ -o $@

test: test_invnorm test_scalerows
	./test_invnorm
	./test_scalerows

clean:
	rm -f test_invnorm test_scalerows
