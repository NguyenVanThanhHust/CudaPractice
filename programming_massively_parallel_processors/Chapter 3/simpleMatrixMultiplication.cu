#include <cuda_runtime.h>
#include <string>
#include <iostream>


using std::cin;
using std::cout;
using std::endl;

#define MATRIX_HEIGHT 100
#define MATRIX_WIDTH 200

int main()
{
    // Host 2d array
    float h_A[MATRIX_WIDTH][MATRIX_HEIGHT];
    float h_B[MATRIX_WIDTH][MATRIX_HEIGHT];
    float h_B[MATRIX_WIDTH][MATRIX_HEIGHT];

    // Check if array is created
    if (h_A==NULL || h_B==NULL || h_C==NULL)
    {
        cout<<"Can't create host array"<<endl;
        exit(EXIT_FAILURE);
    }

    // Initialize host vector
    for (int  row_index = 0; row_index < MATRIX_HEIGHT; row_index++)
    {
        for (int col_index = 0; col_index < MATRIX_WIDTH; col_index++)
        {
            h_A[row_index][col_index] = float(row_index * MATRIX_WIDTH + 1);
            h_B[row_index][col_index] = float(row_index * MATRIX_WIDTH + 2);
        }
    }

    return 0;
}