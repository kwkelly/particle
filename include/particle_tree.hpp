#include <complex>
#include "El.hpp"
#include <cheb_node.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <pvfmm.hpp>
#include <pvfmm_common.hpp>
#include <cassert>
#include <cstring>
#include <profile.hpp>
#include <mpi.h>
#include "helm_kernels.hpp"
#include <set>
#include <funcs.hpp>

#ifndef PARTICLE_TREE_HPP
#define PARTICLE_TREE_HPP

namespace isotree{
class ParticleTree{

	//typedef typename FMM_Mat_t::FMMNode_t FMM_Node_t;
	//typedef typename FMM_Node_t::Node_t Node_t;
	//typedef typename FMM_Mat_t::Real_t Real_t;
	//typedef typename FMM_Node_t::NodeData PtFMM_Data_t;
	//typedef typename pvfmm::FMM_Tree<FMM_Mat_t> PtFMM_Tree_t;

	public:

		enum AVec{
			src,
			trg
		};
		ParticleTree(MPI_Comm c = MPI_COMM_WORLD){};

		ParticleTree(std::vector<double> &src_coord, std::vector<double> &src_value, std::vector<double> &trg_coord, std::vector<double> &pt_coord, pvfmm::BoundaryType bndry = pvfmm::FreeSpace, MPI_Comm comm = MPI_COMM_WORLD);
		ParticleTree(std::vector<double> &src_coord, std::vector<double> &trg_coord, std::vector<double> &pt_coord, pvfmm::BoundaryType bndry = pvfmm::FreeSpace, MPI_Comm comm = MPI_COMM_WORLD);

		~ParticleTree(){
			if(!fmm_mat) delete fmm_mat;
			if(!tree_data) delete tree_data;
			if(!tree) delete tree;
		};


		pvfmm::BoundaryType bndry;

		std::vector<double> pt_coord;
		std::vector<double> src_coord;
		std::vector<double> src_value;
		std::vector<double> trg_coord;
		std::vector<double> trg_value;

		long long m,M,n,N,l,L;
		static int mult_order;
		static bool adap;
		static int maxdepth;
		static int dim;
		static int data_dof;
		static int max_pts;
		MPI_Comm comm;

		void RunFMM();
		void ApplyFn(void (*fn)(const  double* coord, int n, double* out), AVec av);
		void Add(ParticleTree& other, std::complex<double> multiplier, AVec av1, AVec av2);
		void Multiply(ParticleTree& other, AVec av1, AVec av2);
		void Conjugate();
		void ScalarMultiply(std::complex<double> multiplier);
		void Copy(const ParticleTree& other);
		void Zero();
		void SetupFMM(const pvfmm::Kernel<double>* kernel);
		double Norm2(AVec av);
		std::complex<double> Integrate();
		std::vector<double> ChebPoints();

	private:
		pvfmm::PtFMM* fmm_mat = NULL;
		pvfmm::PtFMM_Tree* tree = NULL;
		pvfmm::PtFMM_Data* tree_data = NULL;

};

}

#endif
