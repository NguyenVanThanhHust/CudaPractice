#include "matmul_cpu.h"

void runOnCPU(int *a, int *b, int *c, int m, int n, int p)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < p; col++)
        {
            int temp_value = 0;
            for (int i = 0; i < n; i++)
            {
                temp_value += a[row*n + i] * b[i*p+col];
            }
            c[row*p + col] = temp_value;
        }
    }
}