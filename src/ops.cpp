#include <iostream>
#include <vector>
#include <algorithm>
#include "convert_elemental.hpp"
#include "particle_tree.hpp"
#include "pvfmm.hpp"
#include "El.hpp"
#include "funcs.hpp"
#include "ops.hpp"
#include <mpi.h>

namespace isotree{

Global_to_det_op::Global_to_det_op(std::vector<double> &global_coord, std::vector<double> &det_coord, void (*masking_fn)(const  double* coord, int n, double* out), const pvfmm::Kernel<double>* kernel,pvfmm::BoundaryType bndry, std::string name, MPI_Comm comm)
	: comm(comm), name(name){
	int rank;
	MPI_Comm_rank(comm,&rank);

	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	this->mask = new ParticleTree(global_coord,det_coord,global_coord,bndry,comm);
	this->mask->ApplyFn(masking_fn,sv);
	this->temp = new ParticleTree(global_coord,det_coord,global_coord,bndry,comm);
	temp->SetupFMM(kernel);
}

Global_to_det_op::~Global_to_det_op(){
	delete mask;
	delete temp;
}

void Global_to_det_op::operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y){
	int rank;
	MPI_Comm_rank(comm,&rank);
	// This function simply computes the convolution of G with an input U
	// and then gets only the output at the detector locations
	if(!rank) std::cout << name << std::endl; 

	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	elemental2tree(x,*temp,sv);
	//temp->Multiply(*mask,sv,sv);
	temp->RunFMM();

	tree2elemental(*temp,y,tv);
	return;
}

Det_to_global_op::Det_to_global_op(std::vector<double> &det_coord, std::vector<double> &global_coord, void (*masking_fn)(const  double* coord, int n, double* out), const pvfmm::Kernel<double>* kernel, pvfmm::BoundaryType bndry, std::string name, MPI_Comm comm)
	: comm(comm), name(name){
	int rank;
	MPI_Comm_rank(comm,&rank);
	
	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	this->mask = new ParticleTree(det_coord,global_coord,global_coord,bndry,comm);
	this->mask->ApplyFn(masking_fn,tv);
	this->temp = new ParticleTree(det_coord,global_coord,global_coord,bndry,comm);
	temp->SetupFMM(kernel);
}

Det_to_global_op::~Det_to_global_op(){
	delete mask;
	delete temp;
}

void Det_to_global_op::operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x){

	int rank;
	MPI_Comm_rank(comm,&rank);

	if(!rank) std::cout << name << std::endl; 

	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	elemental2tree(y,*temp,sv);
	temp->RunFMM();
	//temp->Multiply(*mask,tv,tv);

	tree2elemental(*temp,x,tv);

	return;
}

}
