#include "coordinate_dataset.h"
#include "matrix.h"


class CoordinateDataset{
private: 
    size_t batch_size;
    size_t number_of_batches;

    std::vector<Matrix> batches;
    std::vector<Matrix> targets;

public:
    CoordinateDataset(size_t batch_size, size_t, number_of_batches);

    int getNumOfBatches();
    std::vector<Matrix>& getBatches();
    std::vector<Matrix>& getTargets();
};
