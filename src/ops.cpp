#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "convert_elemental.hpp"
#include "particle_tree.hpp"
#include "pvfmm.hpp"
#include "El.hpp"
#include "funcs.hpp"
#include "ops.hpp"
#include <mpi.h>

namespace isotree{

FMM_op::FMM_op(std::vector<double> &src_coord, std::vector<double> &trg_coord, const pvfmm::Kernel<double>* kernel,pvfmm::BoundaryType bndry, std::string name, MPI_Comm comm)
	: comm(comm), name(name){
	int rank;
	MPI_Comm_rank(comm,&rank);

	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	this->temp = new ParticleTree(src_coord,trg_coord,trg_coord,bndry,comm);
	temp->SetupFMM(kernel);
}

FMM_op::~FMM_op(){
	delete temp;
}

void FMM_op::operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y){
	int rank;
	MPI_Comm_rank(comm,&rank);
	// This function simply computes the convolution of G with an input U
	// and then gets only the output at the detector locations
	//if(!rank) std::cout << name << std::endl; 

	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	elemental2tree(x,*temp,sv);
	temp->RunFMM();
	tree2elemental(*temp,y,tv);
	return;
}

B_op::B_op(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &S_G, const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &V_G, const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &U_hat, std::string name, const El::Grid &g, MPI_Comm comm)
	: S_G(S_G), V_G(V_G), U_hat(U_hat){
		R_d = V_G.Width();
		R_s = U_hat.Width();
		y_rect.SetGrid(g);
		El::Zeros(y_rect,R_d,R_s);

	}

B_op::~B_op(){}

void B_op::operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y){
	//std::cout << name <<std::endl;
	auto x_U_hat = U_hat;
	El::DiagonalScale(El::LEFT,El::NORMAL,x,x_U_hat);
	El::Gemm(El::ADJOINT,El::NORMAL,El::Complex<double>(1.0),V_G,x_U_hat,El::Complex<double>(0.0),y_rect);
	El::DiagonalScale(El::LEFT,El::NORMAL,S_G,y_rect);
	El::Zeros(y,R_s*R_d,1);
	for(int i=0;i<R_s;i++){
		const auto y_rect_i = El::LockedView(y_rect,0,i,R_d,1);
		std::vector<int> rows(R_d);
		std::iota(rows.begin(),rows.end(),i*R_d);
		std::vector<int> cols = {0};
		El::SetSubmatrix(y,rows,cols,y_rect_i);
	}
	return;

}

Bt_op::Bt_op(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &S_G, const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &V_G, const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &U_hat, std::string name, const El::Grid &g, MPI_Comm comm)
	: S_G(S_G), V_G(V_G), U_hat(U_hat){
	R_d = V_G.Width();			
	R_s = U_hat.Width();
	N   = V_G.Height();
	x_rect.SetGrid(g);
	temp2.SetGrid(g);
	temp3.SetGrid(g);
	ones.SetGrid(g);
	U_hat_conj.SetGrid(g);
	U_hat_conj = U_hat;
	El::Conjugate(U_hat_conj);
	El::Ones(ones,R_s,1);
	El::Zeros(x_rect,R_d,R_s);
	El::Zeros(temp2,N,R_s);
	El::Zeros(temp3,N,R_s);
	
	}

Bt_op::~Bt_op(){}

void Bt_op::operator()(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &x, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &y){
	//std::cout << name << std::endl;
	for(int i=0;i<R_s;i++){
		const auto x_i = El::LockedView(x,i*R_d,0,R_d,1);
		std::vector<int> rows(R_d);
		std::iota(rows.begin(), rows.end(),0);
		std::vector<int> cols = {i};
		El::SetSubmatrix(x_rect,rows,cols,x_i);
	}
	El::DiagonalScale(El::LEFT,El::NORMAL,S_G,x_rect);
	El::Zeros(y,N,1);
	El::Gemm(El::NORMAL,El::NORMAL,El::Complex<double>(1.0),V_G,x_rect,El::Complex<double>(0.0),temp2);
	El::Hadamard(U_hat_conj,temp2,temp3);
	El::Gemv(El::NORMAL,El::Complex<double>(1.0),temp3,ones,El::Complex<double>(0.0),y);

	return;
}

}
