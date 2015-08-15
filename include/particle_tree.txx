#include <cheb_node.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <fmm_pts.hpp>
#include <pvfmm_common.hpp>
#include <mpi.h>
#include <profile.hpp>
#include <pvfmm.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <iterator>
#include <iomanip>
#include "vec_ops.hpp"

namespace isotree{

int ParticleTree::mult_order;
bool ParticleTree::adap;
int ParticleTree::maxdepth;
int ParticleTree::dim;
int ParticleTree::data_dof;
int ParticleTree::max_pts;


	ParticleTree::ParticleTree(std::vector<double> &src_coord, std::vector<double> src_value, std::vector<double> &trg_coord, std::vector<double> &pt_coord, ParticleTree::AVec avec, pvfmm::BoundaryType bndry, MPI_Comm comm): 
		src_coord(src_coord), src_value(src_value), trg_coord(trg_coord), pt_coord(pt_coord), bndry(bndry), comm(comm){


			int max_depth = ParticleTree::maxdepth; 
			int max_pts = ParticleTree::max_pts; 
			int data_dof = ParticleTree::data_dof; 

			tree_data = new pvfmm::PtFMM_Data;
			bool adap=true;

			tree_data->dim=COORD_DIM;
			tree_data->max_depth=MAX_DEPTH;
			tree_data->max_pts=max_pts;

			std::vector<double> surf_coord;
			std::vector<double> surf_value;
			// Set source points.
			tree_data-> src_coord= src_coord;
			tree_data-> src_value= src_value;
			tree_data->surf_coord=surf_coord;
			tree_data->surf_value=surf_value;

			// Set target points.
			tree_data->trg_coord=trg_coord;
			tree_data-> pt_coord= pt_coord;

			tree=new pvfmm::PtFMM_Tree(comm);
			tree->Initialize(tree_data);
			tree->InitFMM_Tree(adap,bndry);

			m = trg_coord.size()/dim;
			n = src_coord.size()/dim;
			MPI_Allreduce(&m,&M,1,MPI_LONG_LONG_INT,MPI_SUM,comm);
			MPI_Allreduce(&n,&N,1,MPI_LONG_LONG_INT,MPI_SUM,comm);

			if(avec == AVec::src) a_vec = &src_value;
			if(avec == AVec::trg) a_vec = &trg_value;


		};



void ParticleTree::RunFMM(){
	tree->ClearFMMData();
	pvfmm::PtFMM_Evaluate(tree, trg_value, m, &src_value);
	tree->Copy_FMMOutput();
	return;
}


void ParticleTree::SetupFMM(const pvfmm::Kernel<double>* kernel){

	int mult_order = ParticleTree::mult_order;
	fmm_mat = new pvfmm::PtFMM;
	fmm_mat->Initialize(mult_order, comm, kernel);

	tree->SetupFMM(fmm_mat);
	return;
}

void ParticleTree::Add(ParticleTree& other, std::complex<double> multiplier){
	add(*(this->a_vec),*(other.a_vec),multiplier);
	return;
}

void ParticleTree::Multiply(ParticleTree& other){
	multiply(*(this->a_vec),*(other.a_vec));
	return;
}

double ParticleTree::Norm2(){
	return norm2(*(this->a_vec),this->comm);
}


}
