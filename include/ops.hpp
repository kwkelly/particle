#include <mpi.h>
#include "particle_tree.hpp"
#include "pvfmm.hpp"
#include <string>
#include "El.hpp"

#ifndef OPS_HPP
#define OPS_HPP

namespace isotree{

class FMM_op{
  public:
		FMM_op(std::vector<double> &src_coord, std::vector<double> &trg_coord, const pvfmm::Kernel<double> *kernel, pvfmm::BoundaryType bndry, std::string name, MPI_Comm comm = MPI_COMM_WORLD);
		~FMM_op();
		void operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC, El::STAR> &y);
	private:
		ParticleTree* temp = NULL;
		MPI_Comm comm;
		std::string name;
};

class B_op{
  public:
		B_op(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &S_G, const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &V_G, const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &U_hat, std::string name, const El::Grid &g, MPI_Comm comm = MPI_COMM_WORLD);
		~B_op();
		void operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC, El::STAR> &y);
	private:
		MPI_Comm comm;
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> S_G;
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G;
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_hat;
		//El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x_U_hat;
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y_rect;
		int R_d;
		int R_s;
		std::string name;
};

class Bt_op{
  public:
		Bt_op(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &S_G, const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &V_G, const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &U_hat, std::string name, const El::Grid &g, MPI_Comm comm = MPI_COMM_WORLD);
		~Bt_op();
		void operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC, El::STAR> &y);
	private:
		MPI_Comm comm;
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> S_G;
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G;
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_hat;
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x_rect;
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_hat_conj;
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp2;
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp3;
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> ones;
		int R_d;
		int R_s;
		int N;
		std::string name;
};



}

#endif
