## A simple cuBLAS Saxpy example
SAXPY
> y = ax + y

Compile it
```
nvcc example_saxpy.cu -lcublas -o example_saxpy
```

Profile it
```
nvprof ./example_saxpy
```
