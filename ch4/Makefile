CU_APPS=cufft rand open
C_APPS=

OPENCV_INCLUDE=-IC:\tools\opencv\include -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include"
OPENCV_BIN=C:\tools\opencv\x64\vc17
LIB=-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64" -LC:\tools\opencv\x64\vc17\lib

build:open.cu
	nvcc -lcuda -lcudart -diag-suppress 611 ${OPENCV_INCLUDE} ${LIB} -o $(subst .cu,.exe,$^) $^ 
	copy ${OPENCV_BIN}\bin\opencv_world470.dll .\

test:rand.cu
	nvcc -lcuda -lcudart -o $(subst .cu,,$^) $^ 
	#nsys profile --stats=true --trace=cuda,nvtx $(subst .cu,.exe,$^)


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
