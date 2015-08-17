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
#include "vec_ops.hpp"
#include "ops.hpp"
#include "point_distribs.hpp"
#include "helm_kernels.hpp"
#include "convert_elemental.hpp"


	typedef pvfmm::FMM_Node<pvfmm::MPI_Node<double> > FMM_Node_t;
	typedef pvfmm::FMM_Pts<FMM_Node_t>                FMM_Mat_t;
	typedef pvfmm::FMM_Tree<FMM_Mat_t>                PtFMM_Tree;
	typedef FMM_Node_t::NodeData                      FMM_Data_t;

	using namespace isotree;

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

int el_test(MPI_Comm &comm){

	int myrank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;


	// set up trg and src points
	std::vector<double> src_coord = unif_point_distrib(1,0.0,1.0,comm);
	std::vector<double> src_value((src_coord.size()*2)/3,0.5);
	if(src_coord.size() > 0) src_value[1] = 0.0;
	std::vector<double> trg_coord = unif_point_distrib(100,0.0,1.0,comm);
	std::vector<double> pt_coord = trg_coord;
	int n_trg = trg_coord.size()/3;

	const std::complex<double> a(1.0,0.0);

	// make trees
	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	ParticleTree poly(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	ParticleTree zero(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	ParticleTree sol(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	poly.ApplyFn(poly_fn,tv);
	zero.ApplyFn(zero_fn,tv);
	sol.ApplyFn(poly_fn,tv);

	//convert poly to Elemental
	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
	El::Zeros(x,poly.M,1);
	tree2elemental(poly,x,tv);
	elemental2tree(x,zero,tv);

	zero.Add(sol,std::complex<double>(-1.0,0.0),tv,tv);
	double nrm = zero.Norm2(tv)/sol.Norm2(tv);

	std::string name = __func__;
	test_less(1e-6,nrm,name,comm);

	
	return 0;
}


int G_test(MPI_Comm &comm){

	int myrank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;


	// set up trg and src points
	std::vector<double> src_coord = unif_point_distrib(8,0.25,0.75,comm);
	int n_loc_src = src_coord.size()/3;
	std::vector<double> src_value((src_coord.size()*2)/3,0.0);
	std::vector<double> trg_coord = unif_point_distrib(10,0.0,1.0,comm);
	int n_loc_trg = trg_coord.size()/3;
	std::vector<double> pt_coord = src_coord;

	const std::complex<double> a(1.0,0.0);

	// make trees
	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	ParticleTree int_test(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	ParticleTree sol(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	sol.ApplyFn(eight_pt_sol_fn, tv);

	int n_src = int_test.N;
	int n_trg = int_test.M;

	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
	El::Ones(x,n_src,1);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gx(g);
	El::Zeros(Gx,n_trg,1);

	Global_to_det_op G(src_coord,trg_coord,one_fn, kernel, bndry,"G",comm);
	G(x,Gx);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Zeros(y,n_trg,1);
	tree2elemental(sol,y,tv);

	El::Axpy(El::Complex<double>(-1.0),y,Gx);
	double nrm = El::TwoNorm(Gx)/El::TwoNorm(y);

	std::string name = __func__;
	test_less(1e-6,nrm,name,comm);

	
	return 0;
}

int Gt_test(MPI_Comm &comm){

	int myrank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;


	// set up trg and src points
	std::vector<double> src_coord = unif_point_distrib(8,0.25,0.75,comm);
	int n_loc_src = src_coord.size()/3;
	std::vector<double> src_value((src_coord.size()*2)/3,0.0);
	std::vector<double> trg_coord = unif_point_distrib(10,0.0,1.0,comm);
	int n_loc_trg = trg_coord.size()/3;
	std::vector<double> pt_coord = src_coord;

	const std::complex<double> a(1.0,0.0);

	// make trees
	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	ParticleTree int_test(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	ParticleTree sol(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	sol.ApplyFn(eight_pt_conj_sol_fn, tv);

	int n_src = int_test.N;
	int n_trg = int_test.M;

	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
	El::Ones(x,n_src,1);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gtx(g);
	El::Zeros(Gtx,n_trg,1);

	Det_to_global_op Gt(src_coord,trg_coord,one_fn, kernel_conj, bndry,"G*", comm);
	Gt(x,Gtx);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Zeros(y,n_trg,1);
	tree2elemental(sol,y,tv);

	El::Axpy(El::Complex<double>(-1.0),y,Gtx);
	double nrm = El::TwoNorm(Gtx)/El::TwoNorm(y);

	std::string name = __func__;
	test_less(1e-6,nrm,name,comm);

	
	return 0;
}


int GGt_test(MPI_Comm &comm){

	int rank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &rank);

	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;


	// set up trg and src points
	std::vector<double> src_coord = unif_point_distrib(100,0.25,0.75,comm);
	int n_loc_src = src_coord.size()/3;
	std::vector<double> src_value((src_coord.size()*2)/3,0.0);
	std::vector<double> trg_coord = unif_point_distrib(100,0.0,1.0,comm);
	int n_loc_trg = trg_coord.size()/3;
	std::vector<double> pt_coord = src_coord;

	const std::complex<double> a(1.0,0.0);

	// make trees
	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	ParticleTree sol(src_coord,src_value,trg_coord,pt_coord,bndry,comm); // probably won't be used
	sol.ApplyFn(eight_pt_conj_sol_fn, tv);

	int n_src = sol.N;
	int n_trg = sol.M;

	El::Grid g(comm);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
	El::Gaussian(x,n_src,1);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gx(g);
	El::Zeros(Gx,n_trg,1);

	Global_to_det_op G(src_coord,trg_coord,one_fn, kernel, bndry,"G",comm);
	G(x,Gx);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
	El::Gaussian(y,n_trg,1);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gty(g);
	El::Zeros(Gty,n_src,1);

	Det_to_global_op Gt(trg_coord,src_coord,one_fn, kernel_conj, bndry,"G*",comm);
	Gt(y,Gty);

	El::Complex<double> Gxy = El::Dot(Gx,y);
	El::Complex<double> xGty = El::Dot(x,Gty);
	if(!rank) std::cout << "Gxy=" << Gxy << std::endl;
	if(!rank) std::cout << "xGty=" << xGty << std::endl;
 	double diff = El::Abs(Gxy - xGty);

	std::string name = __func__;
	test_less(1e-6,diff,name,comm);

	
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
	std::vector<double> src_value((src_coord.size()*2/3),0.5);
	if(src_coord.size() > 0) src_value[1] = 0.0;
	std::vector<double> trg_coord = unif_point_distrib(100,0.0,1.0,comm);
	std::vector<double> pt_coord = trg_coord;
	int n_trg = trg_coord.size()/3;

	const std::complex<double> a(1.0,0.0);

	isotree::ParticleTree::AVec tv = isotree::ParticleTree::AVec::trg;
	isotree::ParticleTree::AVec sv = isotree::ParticleTree::AVec::src;

	isotree::ParticleTree one(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	isotree::ParticleTree two(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	isotree::ParticleTree sol(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	one.ApplyFn(poly_fn,tv);
	two.ApplyFn(poly_fn,tv);

	sol.ApplyFn(poly_prod_fn,tv);

	one.Multiply(two,tv,tv);

	one.Add(sol,std::complex<double>(-1.0,0.0),tv,tv);
	double nrm = one.Norm2(tv)/sol.Norm2(tv);

	std::string name = __func__;
	test_less(1e-6,nrm,name,comm);

	
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
	std::vector<double> src_value((src_coord.size()*2)/3,0.5);
	if(src_coord.size() > 0 ) src_value[1] = 0.0;
	std::vector<double> trg_coord = unif_point_distrib(100,0.0,1.0,comm);
	std::vector<double> pt_coord = trg_coord;
	int n_trg = trg_coord.size()/3;

	isotree::ParticleTree::AVec tv = isotree::ParticleTree::AVec::trg;
	isotree::ParticleTree::AVec sv = isotree::ParticleTree::AVec::src;
	isotree::ParticleTree one(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	one.SetupFMM(kernel);
	one.RunFMM();

	isotree::ParticleTree two(src_coord,src_value,trg_coord,pt_coord,bndry,comm);
	two.SetupFMM(kernel);
	two.RunFMM();
	const std::complex<double> a = 1.0;
	one.Add(two,a,tv,tv);
	
	std::vector<double> trg = one.trg_value;
	std::vector<double> trg_should_be(trg.size());
	ctr_pt_sol_fn(&trg_coord[0], n_trg, &trg_should_be[0]);
	double snrm = norm2(trg_should_be,comm);

	add(trg_should_be,trg,std::complex<double>(-1.0,0.0));
	double nrm = norm2(trg_should_be,comm);

	std::string name = __func__;
	test_less(1e-6,nrm/snrm,name,comm);

	return 0;
}

int grsvd_test(MPI_Comm &comm){
	int N_d_sugg = 1000;
	int N_s_sugg = 100;
	int R_d = 25;
	int R_s = 25;
	int R_b = 25;
	int k = 25;
	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel_low;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj_low;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	double rsvd_tol = 0.00001;

	int data_dof = 2;

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////
	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;

	std::vector<double> pt_srcs = unif_plane(N_s_sugg,0,0.1,comm);

	int lN_s = pt_srcs.size()/3;
	int N_s;
	MPI_Allreduce(&lN_s,&N_s,1,MPI_INT,MPI_SUM,comm);
	if(!rank) std::cout << "Number of sources generated=" << N_s << std::endl;


	//std::vector<double> d_locs = unif_plane(N_d_sugg, 0, 0.9, comm);
	std::vector<double> d_locs = rand_sph_point_distrib(N_d_sugg, 0.45, comm);

	int lN_d = d_locs.size()/3;
	int N_d;
	MPI_Allreduce(&lN_d, &N_d, 1, MPI_INT, MPI_SUM, comm);
	if(!rank) std::cout << "Number of detectors generated=" << N_d << std::endl;

	std::vector<double> ev_coord = unif_point_distrib(10000,0.0,1.0,comm);

	// also needs to be global...

	//////////////////////////////////////////////////////////////
	// File name prefix
	//////////////////////////////////////////////////////////////
	std::string params = "-" + std::to_string((long long)(ParticleTree::maxdepth)) + "-" + std::to_string((long long)(N_d)) + "-" + std::to_string((long long)(N_s)) + "-" + std::to_string((long long)(R_d)) + "-" + std::to_string((long long)(R_s)) + "-" + std::to_string((long long)(R_b)) + "-" + std::to_string((long long)(k));
	if(!rank) std::cout << "Params " << params << std::endl;


	//////////////////////////////////////////////////////////////
	// Set up FMM stuff
	//////////////////////////////////////////////////////////////

	ParticleTree temp(ev_coord,d_locs,ev_coord,bndry,comm);
	ParticleTree mask(ev_coord,d_locs,ev_coord,bndry,comm);
	mask.ApplyFn(mask_fn,sv);

	// initialize the trees

	// Tree sizes
	int m = temp.m;
	int M = temp.M; //m is small, the number of detectors
	int n = temp.n;
	int N = temp.N; // n is the discretization of the whole space
	int N_disc = N;

	// Set the scalars for multiplying in Gemm
	auto alpha = El::Complex<double>(1.0);
	auto beta = El::Complex<double>(0.0);

	// Set grid
	El::Grid g(comm);

	/////////////////////////////////////////////////////////////////
	// Low rank factorization of G
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Low Rank Factorization of G" << std::endl;

	Global_to_det_op G(ev_coord,d_locs,mask_fn, kernel, bndry,"G",comm);
	Det_to_global_op Gt(d_locs,ev_coord,mask_fn, kernel_conj, bndry,"G*",comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_G(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR>	S_G(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G(g);

	rsvd::RSVDCtrl ctrl_G;
	ctrl_G.m=N_d;
	ctrl_G.n=N_disc;
	ctrl_G.r=R_d;
	ctrl_G.l=10;
	ctrl_G.q=0;
	ctrl_G.tol=rsvd_tol;
	ctrl_G.max_sz=N_d;
	ctrl_G.adap=rsvd::ADAP;
	ctrl_G.orientation=rsvd::NORMAL;
	rsvd::rsvd(U_G,S_G,V_G,G,Gt,ctrl_G);
	R_d = ctrl_G.r;
	if(!rank) std::cout << "R_d = " << R_d << std::endl;

	El::Write(U_G,SAVE_DIR_STR+"U_G"+params,El::ASCII_MATLAB);
	El::Write(S_G,SAVE_DIR_STR+"S_G"+params,El::ASCII_MATLAB);
	El::Write(V_G,SAVE_DIR_STR+"V_G"+params,El::ASCII_MATLAB);

	{ // Test that G is good

		/*
		 * See if the ||Gw - USV*w||/||Gw|| is small
		 */

		{
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> r(g);
			El::Gaussian(r,N_disc,1);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> e(g);
			El::Zeros(e,R_d,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,e);
			El::DiagonalScale(El::LEFT,El::NORMAL,S_G,e);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> g1(g);
			El::Zeros(g1,N_d,1);
			El::Gemm(El::NORMAL,El::NORMAL,alpha,U_G,e,beta,g1); // GY now actually U

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> i(g);
			El::Zeros(i,N_d,1);

			G(r,i);
			//El::Display(i,"i");
			//El::Display(g1,"g1");
			//elemental2tree(i,&temp);
			//temp.Add(&w,-1.0);

			El::Axpy(-1.0,i,g1);
			double ndiff = El::TwoNorm(g1)/El::TwoNorm(i);
			if(!rank) std::cout << "||Gr - USV*r|/||Gr|||=" << ndiff << std::endl;
		}


		/*
		 * See if the ||G*r - VSU*r||/||G*r|| is small
		 */

		{
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> r(g);
			El::Gaussian(r,N_d,1);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> e(g);
			El::Zeros(e,R_d,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,U_G,r,beta,e);
			El::DiagonalScale(El::LEFT,El::NORMAL,S_G,e);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> g1(g);
			El::Zeros(g1,N_disc,1);
			El::Gemm(El::NORMAL,El::NORMAL,alpha,V_G,e,beta,g1); // GY now actually U

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> i(g);
			El::Zeros(i,N_disc,1);

			Gt(r,i);
			//El::Display(i,"i");
			//El::Display(g1,"g1");
			//elemental2tree(i,&temp);
			//temp.Add(&w,-1.0);

			El::Axpy(-1.0,i,g1);
			double ndiff = El::TwoNorm(g1)/El::TwoNorm(i);
			if(!rank) std::cout << "||G*r - VSU*r|/||G*r|||=" << ndiff << std::endl;
		}



		/*
		 * Test that ||r - VV*r||/||r|| is small so that w can be well represented in the basis of V
		 */
		{
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> r(g);
			El::Gaussian(r,N_disc,1);
			elemental2tree(r,temp,sv);
			temp.Multiply(mask,sv,sv);
			tree2elemental(temp,r,sv);
			El::Display(r,"r");

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vt_Gr(g);
			El::Zeros(Vt_Gr,R_d,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,Vt_Gr);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VVt_Gr(g);
			El::Zeros(VVt_Gr,N_disc,1);
			El::Gemm(El::NORMAL,El::NORMAL,alpha,V_G,Vt_Gr,beta,VVt_Gr);
			El::Display(VVt_Gr,"VV*r");

			El::Axpy(-1.0,r,VVt_Gr);
			double coeff_relnorm = El::TwoNorm(VVt_Gr)/El::TwoNorm(r);
			if(!rank) std::cout << "||r - VV*r||/||r||=" << coeff_relnorm << std::endl;
		}

		/*
		 * Take a look at the first and last few vectors that for the basis for V and ensure that the columns are orthogonal
		 */

		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_1 = El::LockedView(V_G,0,0,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_2 = El::LockedView(V_G,0,1,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_3 = El::LockedView(V_G,0,2,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_4 = El::LockedView(V_G,0,3,N_disc,1);

		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_l1 = El::LockedView(V_G,0,R_d-1,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_l2 = El::LockedView(V_G,0,R_d-2,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_l3 = El::LockedView(V_G,0,R_d-3,N_disc,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_l4 = El::LockedView(V_G,0,R_d-4,N_disc,1);
		/*
			 elemental2tree(V_G_1,&temp);
			 temp.Write2File((SAVE_DIR_STR+"v1").c_str(),VTK_ORDER);
			 elemental2tree(V_G_2,&temp);
			 temp.Write2File((SAVE_DIR_STR+"v2").c_str(),VTK_ORDER);
			 elemental2tree(V_G_3,&temp);
			 temp.Write2File((SAVE_DIR_STR+"v3").c_str(),VTK_ORDER);
			 elemental2tree(V_G_4,&temp);
			 temp.Write2File((SAVE_DIR_STR+"v4").c_str(),VTK_ORDER);
			 elemental2tree(V_G_l1,&temp);
			 temp.Write2File((SAVE_DIR_STR+"vl1").c_str(),VTK_ORDER);
			 elemental2tree(V_G_l2,&temp);
			 temp.Write2File((SAVE_DIR_STR+"vl2").c_str(),VTK_ORDER);
			 elemental2tree(V_G_l3,&temp);
			 temp.Write2File((SAVE_DIR_STR+"vl3").c_str(),VTK_ORDER);
			 elemental2tree(V_G_l4,&temp);
			 temp.Write2File((SAVE_DIR_STR+"vl4").c_str(),VTK_ORDER);
			 */


		/*
		// test that the vectors are orhtogonal
		elemental2tree(V_G_1,&temp);
		elemental2tree(V_G_2,&temp2);
		temp.ConjMultiply(&temp2,1);
		std::vector<double> integral = temp.Integrate();
		if(!rank) std::cout << "<V_G_1, V_G_2> = " << integral[0] << "+i" << integral[1] << std::endl;

		elemental2tree(V_G_1,&temp);
		elemental2tree(V_G_3,&temp2);
		temp.ConjMultiply(&temp2,1);
		integral = temp.Integrate();
		if(!rank) std::cout << "<V_G_1, V_G_3> = " << integral[0] << "+i" << integral[1] << std::endl;

		elemental2tree(V_G_1,&temp);
		elemental2tree(V_G_l1,&temp2);
		temp.ConjMultiply(&temp2,1);
		integral = temp.Integrate();
		if(!rank) std::cout << "<V_G_1, V_G_end> = " << integral[0] << "+i" << integral[1] << std::endl;
		*/

		/*
		 * Test the orthogonality of the matrices via elemental
		 */

		{
			El::DistMatrix<El::Complex<double>,VR,STAR> I(g);
			El::DistMatrix<El::Complex<double>,VR,STAR> UtU(g);
			El::Zeros(UtU,R_d,R_d);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,U_G,U_G,beta,UtU);
			El::Identity(I,R_d,R_d);
			El::Axpy(-1.0,I,UtU);
			double ortho_diff = El::TwoNorm(UtU)/El::TwoNorm(I);
			if(!rank) std::cout << "Ortho diff U: " << ortho_diff << std::endl;

			El::DistMatrix<El::Complex<double>,VR,STAR> VtV(g);
			El::Zeros(VtV,R_d,R_d);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,V_G,beta,VtV);
			El::Axpy(-1.0,I,VtV);
			ortho_diff = El::TwoNorm(VtV)/El::TwoNorm(I);
			if(!rank) std::cout << "Ortho diff V: " << ortho_diff << std::endl;
		}

		/*
		 * Take a look at the spectrum
		 */

		El::Display(S_G,"Sigma_G");

		/*
		 * s_1*||V*w||
		 */

		{
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> r(g);
			El::Gaussian(r,N_disc,1);
			double sig_1 = El::RealPart(S_G.Get(0,0));
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vtr(g);
			El::Zeros(Vtr,R_d,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,Vtr);
			double Vtr_norm = El::TwoNorm(Vtr);

			if(!rank) std::cout << "s_1 * ||V*r||=" << sig_1*Vtr_norm << std::endl;
		}

		/*
		 * ||G_v1 - s_1 u_1||
		 */
		{
			const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_1 = El::LockedView(V_G,0,0,N_disc,1);
			double sig_1 = El::RealPart(S_G.Get(0,0));
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gv1(g);
			El::Zeros(Gv1,N_d,1);
			G(V_G_1,Gv1);

			const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_G_1 = El::LockedView(U_G,0,0,N_d,1);
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp_vec(g);
			El::Zeros(temp_vec,N_d,1);
			El::Axpy(sig_1,U_G_1,temp_vec);
			El::Axpy(-1.0,Gv1,temp_vec);
			double norm_diff = El::TwoNorm(temp_vec);

			if(!rank) std::cout << "||Gv_1 - s_1 u_1||=" << norm_diff << std::endl;
		}

		/*
		 * G*u_1 - s_1 v_1||
		 */
		{
			double sig_1 = El::RealPart(S_G.Get(0,0));
			const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_G_1 = El::LockedView(U_G,0,0,N_d,1);
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp_vec(g);
			El::Zeros(temp_vec,N_disc,1);
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gtu1(g);
			El::Zeros(Gtu1,N_disc,1);
			Gt(U_G_1,Gtu1);

			//const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_0 = El::LockedView(V_G,0,0,N_disc,1);
			//elemental2tree(Gtu1,&temp);
			//temp.Write2File((SAVE_DIR_STR+"Gtu1").c_str(),VTK_ORDER);
			El::Axpy(sig_1,V_G_1,temp_vec);
			El::Axpy(-1.0,Gtu1,temp_vec);
			double norm_diff = El::TwoNorm(temp_vec);

			if(!rank) std::cout << "||G*u_1 - s_1 v_1||=" << norm_diff << std::endl;
		}

		/*
		 * ensure that ||G eta|| <=s1*||V'eta||
		 */
		{
			double sig_1 = El::RealPart(S_G.Get(0,0));
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> r(g);
			El::Gaussian(r,N_disc,1);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gr2(g);
			El::Zeros(Gr2,N_d,1);
			G(r,Gr2);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vt_Gr2(g);
			El::Zeros(Vt_Gr2,R_d,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,Vt_Gr2);

			double g_r_norm2 = El::TwoNorm(Gr2);
			double Vt_Gr_norm = El::TwoNorm(Vt_Gr2);
			if(!rank) std::cout << "||Gw|| <= s_1*||V*w||=" << (g_r_norm2 <= sig_1*Vt_Gr_norm) << std::endl;
		}
	}

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
	el_test(comm);         MPI_Barrier(comm);
	add_test(comm);        MPI_Barrier(comm);
	mult_test(comm);       MPI_Barrier(comm);
	G_test(comm);          MPI_Barrier(comm);
	Gt_test(comm);         MPI_Barrier(comm);
	GGt_test(comm);        MPI_Barrier(comm);
	grsvd_test(comm);      MPI_Barrier(comm);

	El::Finalize();

	return 0;
}
