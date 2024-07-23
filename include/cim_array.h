// include/cim_array.h
#ifndef CIM_ARRAY_H
#define CIM_ARRAY_H

#include <vector>

class CIMArray {
public:
    CIMArray(int size);
    void loadMatrix(const std::vector<std::vector<float>>& matrix);
    void loadVector(const std::vector<float>& vector);
    std::vector<float> multiply() const;

private:
    int size_;
    std::vector<std::vector<float>> matrix_;
    std::vector<float> vector_;
};

#endif // CIM_ARRAY_H
