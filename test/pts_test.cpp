#include "particle_tree.hpp"
#include <iostream>
#include <profile.hpp>
#include "funcs.hpp"
#include <pvfmm.hpp>
#include <ctime>
#include <string>
#include <random>
#include "El.hpp"
#include "rsvd.hpp"
#include "point_distribs.hpp"
#include "helm_kernels.hpp"


	typedef pvfmm::FMM_Node<pvfmm::MPI_Node<double> > FMM_Node_t;
	typedef pvfmm::FMM_Pts<FMM_Node_t>                FMM_Mat_t;
	typedef pvfmm::FMM_Tree<FMM_Mat_t>                PtFMM_Tree;
	typedef FMM_Node_t::NodeData                      FMM_Data_t;

#define VTK_ORDER 4
std::string SAVE_DIR_STR;


int test_less(const double &expected, const double &actual, const std::string &name, MPI_Comm &comm){
	int rank;
	MPI_Comm_rank(comm, &rank);
	if(rank == 0){
		if(actual < expected) std::cout << "\033[2;32m" << name << " passed! \033[0m- relative error=" << actual  << " expected=" << expected << std::endl;
		else std::cout << "\033[2;31m FAILURE! - " << name << " failed! \033[0m- relative error=" << actual << " expected=" << expected << std::endl;
	}
	return 0;
}

int mult_test(MPI_Comm &comm){

	int myrank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	std::vector<double> src_coord = unif_point_distrib(1,0.0,1.0,comm);
	std::vector<double> src_value(src_coord.size()*2/3,0.5);
	src_value[1] = 0.0;
	std::vector<double> trg_coord = unif_point_distrib(100,0.0,1.0,comm);
	std::vector<double> pt_coord = trg_coord;
	int n_trg = trg_coord.size()/3;

	isotree::ParticleTree::AVec avec = isotree::ParticleTree::AVec::trg;
	isotree::ParticleTree one(src_coord,src_value,trg_coord,pt_coord,avec,bndry,comm);
	one.SetupFMM(kernel);
	one.RunFMM();

	isotree::ParticleTree two(src_coord,src_value,trg_coord,pt_coord,avec,bndry,comm);
	two.SetupFMM(kernel);
	two.RunFMM();
	const std::complex<double> a = 1.0;
	one.Add(two,a);
	
	std::vector<double> trg = one.trg_value;
	std::vector<double> trg_should_be(trg.size());
	ctr_pt_sol_fn(&trg_coord[0], n_trg, &trg_should_be[0]);

	std::cout << "diff" << std::endl;
	for(int i=0; i<trg.size();i++){
		std::cout << trg_should_be[i] - trg[i]<< std::endl;
	}

	return 0;
}


int add_test(MPI_Comm &comm){

	int myrank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;

	std::vector<double> src_coord = unif_point_distrib(1,0.0,1.0,comm);
	std::vector<double> src_value(src_coord.size()*2/3,0.5);
	src_value[1] = 0.0;
	std::vector<double> trg_coord = unif_point_distrib(100,0.0,1.0,comm);
	std::vector<double> pt_coord = trg_coord;
	int n_trg = trg_coord.size()/3;

	isotree::ParticleTree::AVec avec = isotree::ParticleTree::AVec::trg;
	isotree::ParticleTree one(src_coord,src_value,trg_coord,pt_coord,avec,bndry,comm);
	one.SetupFMM(kernel);
	one.RunFMM();

	isotree::ParticleTree two(src_coord,src_value,trg_coord,pt_coord,avec,bndry,comm);
	two.SetupFMM(kernel);
	two.RunFMM();
	const std::complex<double> a = 1.0;
	one.Add(two,a);
	
	std::vector<double> trg = one.trg_value;
	std::vector<double> trg_should_be(trg.size());
	ctr_pt_sol_fn(&trg_coord[0], n_trg, &trg_should_be[0]);

	return 0;
}


////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]){
	static char help[] = "\n\
												-ref_tol    <Real>   Tree refinement tolerance\n\
												-min_depth  <Int>    Minimum tree depth\n\
												-max_depth  <Int>    Maximum tree depth\n\
												-fmm_m      <Int>    Multipole order (+ve even integer)\n\
												-dir        <String> Direcory for saming matrices to\n\
												";

	El::Initialize( argc, argv );
	//pvfmm::Profile::Enable(true);

	MPI_Comm comm=MPI_COMM_WORLD;
	int    rank,size;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&size);


	///////////////////////////////////////////////
	// SETUP
	//////////////////////////////////////////////

	isotree::ParticleTree::dim = 3;
	isotree::ParticleTree::data_dof = 2;
	isotree::ParticleTree::max_pts = 100;
	isotree::ParticleTree::mult_order = El::Input("-fmm_m","Multipole order",10);
	isotree::ParticleTree::maxdepth = El::Input("-max_depth","Maximum tree depth",3);
	isotree::ParticleTree::adap = El::Input("-adap","Adaptivity for tree construction",true);
	SAVE_DIR_STR = El::Input("-dir","Directory for saving the functions and the matrices to",".");
	if(!rank) std::cout << SAVE_DIR_STR << std::endl;

	///////////////////////////////////////////////
	// TESTS
	//////////////////////////////////////////////
	add_test(comm);          MPI_Barrier(comm);
	//mult_test(comm);         MPI_Barrier(comm);

	El::Finalize();

	return 0;
}
