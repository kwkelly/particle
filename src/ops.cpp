#include <iostream>
#include <vector>
#include <algorithm>
#include "convert_elemental.hpp"
#include "invmed_tree.hpp"
#include "pvfmm.hpp"
#include "El.hpp"
#include "funcs.hpp"
#include "ops.hpp"
#include <mpi.h>

namespace isotree{

typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<double> > FMMNode_t;
typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;

Volume_FMM_op::Volume_FMM_op(std::vector<double> detector_coord, void (*masking_fn)(const  double* coord, int n, double* out), const pvfmm::Kernel<double>* kernel,pvfmm::BoundaryType bndry, MPI_Comm comm)
	: det_coord(detector_coord), comm(comm), bndry(bndry){
	int rank;
	MPI_Comm_rank(comm,&rank);

	// first we make sure that we create the mask tree
	this->mask = new ChebTree<FMM_Mat_t>(masking_fn,1.0,kernel,bndry,comm);
	mask->CreateTree(false);
	// create the temporary tree
	this->temp = new ChebTree<FMM_Mat_t>(zero_fn,1.0,kernel,bndry,comm);
	temp->CreateTree(false);

	temp->FMMSetup();
}

Volume_FMM_op::~Volume_FMM_op(){
	delete mask;
	delete temp;
}

void Volume_FMM_op::operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y){
	int rank;
	MPI_Comm_rank(comm,&rank);
	// This function simply computes the convolution of G with an input U
	// and then gets only the output at the detector locations
	if(!rank) std::cout << "VolumeFMM" << std::endl; 

	elemental2tree(x,*temp);
	temp->Multiply(*mask,1);
	temp->ClearFMMData();
	temp->RunFMM();
	temp->Copy_FMMOutput();

	std::vector<double> detector_values = temp->ReadVals(det_coord);

	vec2elemental(detector_values,y);
	return;
}

Particle_FMM_op::Particle_FMM_op(std::vector<double> detector_coord, void (*masking_fn)(const  double* coord, int n, double* out), const pvfmm::Kernel<double>* kernel, MPI_Comm comm)
	: detector_coord(detector_coord), comm(comm){
	int rank;
	MPI_Comm_rank(comm,&rank);
	// first we make sure that we create the mask tree
	this->mask = new ChebTree<FMM_Mat_t>(masking_fn,1.0,kernel,bndry,comm);
	mask->CreateTree(false);
	// create the temporary tree
	this->temp = new ChebTree<FMM_Mat_t>(zero_fn,1.0,kernel,bndry,comm);
	temp->CreateTree(false);

	local_cheb_points = (temp->ChebPoints()).size()/3;
	int detector_value_size = detector_coord.size()*2/3;
	detector_values.resize(detector_value_size);
	std::fill(detector_values.begin(), detector_values.end(), 0.0);

	int mult_order = ChebTree<FMM_Mat_t>::mult_order;

	this->trg_coord = temp->ChebPoints();
	this->local_cheb_points = trg_coord.size()/3;

	// Now we can create the new octree
	this->pt_tree=pvfmm::PtFMM_CreateTree(detector_coord, detector_values, trg_coord, comm );
	// Load matrices.
	matrices = new pvfmm::PtFMM;
	matrices->Initialize(mult_order, comm, kernel);

	// FMM Setup
	pt_tree->SetupFMM(matrices);

}

Particle_FMM_op::~Particle_FMM_op(){
	delete mask;
	delete temp;
	delete pt_tree;
	delete matrices;
}

void Particle_FMM_op::operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x){

	int rank;
	MPI_Comm_rank(comm,&rank);

	if(!rank) std::cout << "ParticleFMM" << std::endl; 

	elemental2vec(y,detector_values);
	pt_tree->ClearFMMData();
	std::vector<double> trg_value;
	pvfmm::PtFMM_Evaluate(pt_tree, trg_value, local_cheb_points, &detector_values);
	//pt_tree->Copy_FMMOutput();
	//pt_tree->Write2File("ptfmm_test",4);
	// Insert the values back in
	temp->Trg2Tree(trg_value);
	temp->Write2File("../results/gt",6);
	temp->Multiply(*mask,1);
	temp->Write2File("../results/gtm",6);
	tree2elemental(*temp, x);

	return;
}

}
