#include "add_function.h"

void addCpu(int* a, int* b, int* c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}