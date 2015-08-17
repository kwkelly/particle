#include <vector>
#include "mpi.h"
#include <complex>

#ifndef VEC_OPS_HPP
#define VEC_OPS_HPP

	//y = ax+y
template<typename T>
void add(std::vector<T> &y, std::vector<T> &x, std::complex<T> a);

// y= x*y
template<typename T>
void multiply(std::vector<T> &y, const std::vector<T> &x);

//y = a*y
template<typename T>
void scalar_multiply(std::vector<T> &y, const std::complex<T> a);

double norm2(std::vector<double> &y, MPI_Comm comm);

#endif
