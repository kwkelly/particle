#include <mpi.h>
#include "particle_tree.hpp"
#include "pvfmm.hpp"
#include <string>
#include "El.hpp"

#ifndef OPS_HPP
#define OPS_HPP

namespace isotree{

class Global_to_det_op{
  public:
		Global_to_det_op(std::vector<double> &global_coord, std::vector<double> &det_coord, void (*masking_fn)(const  double* coord, int n, double* out), const pvfmm::Kernel<double> *kernel, pvfmm::BoundaryType bndry, std::string name, MPI_Comm comm = MPI_COMM_WORLD);
		~Global_to_det_op();
		void operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC, El::STAR> &y);
	private:
		ParticleTree* mask = NULL;
		ParticleTree* temp = NULL;
		MPI_Comm comm;
		std::string name;
};

class Det_to_global_op{
  public:
		Det_to_global_op(std::vector<double> &det_coord, std::vector<double> &global_coord, void (*masking_fn)(const  double* coord, int n, double* out), const pvfmm::Kernel<double> *kernel, pvfmm::BoundaryType bndry, std::string name,  MPI_Comm comm = MPI_COMM_WORLD);
		~Det_to_global_op();
		void operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC, El::STAR> &x);
	private:
		ParticleTree* mask = NULL;
		ParticleTree* temp = NULL;
		MPI_Comm comm;
		std::string name;
};

}

#endif
