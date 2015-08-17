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
#include "particle_tree.hpp"
#include "vec_ops.hpp"

namespace isotree{

int ParticleTree::mult_order;
bool ParticleTree::adap;
int ParticleTree::maxdepth;
int ParticleTree::dim;
int ParticleTree::data_dof;
int ParticleTree::max_pts;


	ParticleTree::ParticleTree(std::vector<double> &src_coord, std::vector<double> &src_value, std::vector<double> &trg_coord, std::vector<double> &pt_coord, pvfmm::BoundaryType bndry, MPI_Comm comm): 
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

			trg_value.resize(m*2);
			MPI_Allreduce(&m,&M,1,MPI_LONG_LONG_INT,MPI_SUM,comm);
			MPI_Allreduce(&n,&N,1,MPI_LONG_LONG_INT,MPI_SUM,comm);

		};


	ParticleTree::ParticleTree(std::vector<double> &src_coord, std::vector<double> &trg_coord, std::vector<double> &pt_coord, pvfmm::BoundaryType bndry, MPI_Comm comm): 
		src_coord(src_coord), trg_coord(trg_coord), pt_coord(pt_coord), bndry(bndry), comm(comm){


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
			src_value = std::vector<double>((src_coord.size()*2)/3,0.0);
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

			trg_value.resize(m*2);
			MPI_Allreduce(&m,&M,1,MPI_LONG_LONG_INT,MPI_SUM,comm);
			MPI_Allreduce(&n,&N,1,MPI_LONG_LONG_INT,MPI_SUM,comm);

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

void ParticleTree::ApplyFn(void (*fn)(const  double* coord, int n, double* out), ParticleTree::AVec av){
	if(av == AVec::trg) fn(&(this->trg_coord[0]),this->m,&(this->trg_value[0]));
	if(av == AVec::src) fn(&(this->src_coord[0]),this->n,&(this->src_value[0]));
	return;
}

void ParticleTree::Add(ParticleTree& other, std::complex<double> multiplier, ParticleTree::AVec av1, ParticleTree::AVec av2){
	if(av1 == AVec::trg and av2 == AVec::trg) add(this->trg_value,other.trg_value,multiplier);
	if(av1 == AVec::src and av2 == AVec::trg) add(this->src_value,other.trg_value,multiplier);
	if(av1 == AVec::trg and av2 == AVec::src) add(this->trg_value,other.src_value,multiplier);
	if(av1 == AVec::src and av2 == AVec::src) add(this->src_value,other.src_value,multiplier);
	return;
}

void ParticleTree::Multiply(ParticleTree& other, ParticleTree::AVec av1, ParticleTree::AVec av2){
	if(av1 == AVec::trg and av2 == AVec::trg) multiply(this->trg_value,other.trg_value);
	if(av1 == AVec::src and av2 == AVec::trg) multiply(this->src_value,other.trg_value);
	if(av1 == AVec::trg and av2 == AVec::src) multiply(this->trg_value,other.src_value);
	if(av1 == AVec::src and av2 == AVec::src) multiply(this->src_value,other.src_value);
	return;
}

double ParticleTree::Norm2(ParticleTree::AVec av){
	if(av == AVec::trg) return norm2(this->trg_value,this->comm);
	if(av == AVec::src) return norm2(this->src_value,this->comm);
}


}
