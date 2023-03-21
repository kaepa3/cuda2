CU_APPS=single
C_APPS=singlecpu

build:single.cu
	nvcc -lcuda -lcudart -o $(subst .cu,,$^) $^ 
	nsys profile --stats=true --trace=cuda,nvtx $(subst .cu,.exe,$^)


all: ${C_APPS} ${CU_APPS}
	echo "all star"
%: %.cu
	nvcc -lcuda -lcudart -o $@ $< 
%: %.c
	clang -o $@.exe $<

clean:
	del *.exe
	del *.exp
	del *.lib

