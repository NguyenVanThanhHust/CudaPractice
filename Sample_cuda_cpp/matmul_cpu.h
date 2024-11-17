#ifndef MATMUL_H
#define MATMUL_H

// Temporay, explicit overwrite of runOnCPU for int and float types
void runOnCPU(int *a, int *b, int *c, int m, int n, int p);
void runOnCPU(float *a, float *b, float *c, int m, int n, int p);

#endif //MATMUL_H
