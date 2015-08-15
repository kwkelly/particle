#include <vector>
#include <complex>
#include <omp.h>

	//y = ax+y
template<typename T>
void add(std::vector<T> &y, std::vector<T> &x, std::complex<T> a){
	assert(y.size() == x.size() and x.size()%2 == 0);

	#pragma omp parallel for
	for(int i=0;i<y.size()/2;i++){
		std::complex<T> yi;
		std::complex<T> xi;
		yi.real(y[2*i+0]);
		yi.imag(y[2*i+1]);

		xi.real(x[2*i+0]);
		xi.imag(x[2*i+1]);

		yi = a*xi+yi;

		y[2*i+0] = yi.real();
		y[2*i+1] = yi.imag();
	}
	return;
}

// y= x*y
template<typename T>
void multiply(std::vector<T> &y, const std::vector<T> &x){
	assert(y.size() == x.size() and x.size()%2 == 0);

	#pragma omp parallel for
	for(int i=0;i<y.size()/2;i++){
		std::complex<T> yi;
		std::complex<T> xi;

		yi.real(y[2*i+0]);
		yi.imag(y[2*i+1]);

		xi.real(x[2*i+0]);
		xi.imag(x[2*i+1]);

		yi = yi*xi;

		y[2*i+0] = yi.real();
		y[2*i+1] = yi.imag();
	}
	return;
}

//y = a*y
template<typename T>
void scalar_multiply(std::vector<T> &y, const std::complex<T> a){
	assert(y.size()%2 == 0);

	#pragma omp parallel for
	for(int i=0;i<y.size()/2;i++){
		std::complex<T> yi;

		yi.real(y[2*i+0]);
		yi.imag(y[2*i+1]);

		yi = a*yi;

		y[2*i+0] = yi.real();
		y[2*i+1] = yi.imag();
	}
	return;
}

// ||y||
double norm2(std::vector<double> &y, MPI_Comm comm){
	assert(y.size()%2 == 0);

	double local_norm = 0.0;
	double norm = 0.0;
	#pragma omp parallel for reduction(+:local_norm)
	for(int i=0;i<y.size()/2;i++){
		std::complex<double> yi;

		yi.real(y[2*i+0]);
		yi.imag(y[2*i+1]);

		local_norm = local_norm + std::real(yi*std::conj(yi));
	}
	MPI_Allreduce(&local_norm,&norm,1,MPI_DOUBLE,MPI_SUM,comm);
	return norm;
}
