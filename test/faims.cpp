
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

template <typename T1, typename T2>
int svd_test(T1 &A, T2 &At, const El::DistMatrix<El::Complex<double>> &U, const El::DistMatrix<El::Complex<double>> &S, const El::DistMatrix<El::Complex<double>> &V, int m, int n, int r, const El::Grid &g, MPI_Comm comm){
	{ // Test that the SVD is good
		int rank, size;
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);
		El::Complex<double> alpha(1.0);
		El::Complex<double> beta(0.0);

		/*
		 * See if the ||Aw - USV*w||/||Aw|| is small
		 */

		{
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> rand(g);
			El::Gaussian(rand,n,1);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> e(g);
			El::Zeros(e,r,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V,rand,beta,e);
			El::DiagonalScale(El::LEFT,El::NORMAL,S,e);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> g1(g);
			El::Zeros(g1,m,1);
			El::Gemm(El::NORMAL,El::NORMAL,alpha,U,e,beta,g1); // GY now actually U

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> i(g);
			El::Zeros(i,m,1);

			A(rand,i);
			//El::Display(i,"i");
			//El::Display(g1,"g1");
			//elemental2tree(i,&temp);
			//temp.Add(&w,-1.0);

			El::Axpy(-1.0,i,g1);
			double ndiff = El::TwoNorm(g1)/El::TwoNorm(i);
			if(!rank) std::cout << "||Ar - USV*r|/||Ar|||=" << ndiff << std::endl;
		}


		/*
		 * See if the ||G*r - VSU*r||/||G*r|| is small
		 */

		{
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> rand(g);
			El::Gaussian(rand,m,1);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> e(g);
			El::Zeros(e,r,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,U,rand,beta,e);
			El::DiagonalScale(El::LEFT,El::NORMAL,S,e);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> g1(g);
			El::Zeros(g1,n,1);
			El::Gemm(El::NORMAL,El::NORMAL,alpha,V,e,beta,g1); // GY now actually U

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> i(g);
			El::Zeros(i,n,1);

			At(rand,i);
			//El::Display(i,"i");
			//El::Display(g1,"g1");
			//elemental2tree(i,&temp);
			//temp.Add(&w,-1.0);

			El::Axpy(-1.0,i,g1);
			double ndiff = El::TwoNorm(g1)/El::TwoNorm(i);
			if(!rank) std::cout << "||A*r - VSU*r|/||A*r|||=" << ndiff << std::endl;
		}



		/*
		 * Test that ||r - VV*r||/||r|| is small so that w can be well represented in the basis of V
		 */
		{
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> rand(g);
			El::Gaussian(rand,n,1);
			//elemental2tree(r,temp,sv);
			//temp.Multiply(mask,sv,sv);
			//tree2elemental(temp,r,sv);
			//El::Display(r,"r");

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vt_Gr(g);
			El::Zeros(Vt_Gr,r,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V,rand,beta,Vt_Gr);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VVt_Gr(g);
			El::Zeros(VVt_Gr,n,1);
			El::Gemm(El::NORMAL,El::NORMAL,alpha,V,Vt_Gr,beta,VVt_Gr);
			//El::Display(VVt_Gr,"VV*r");

			El::Axpy(-1.0,rand,VVt_Gr);
			double coeff_relnorm = El::TwoNorm(VVt_Gr)/El::TwoNorm(rand);
			if(!rank) std::cout << "||r - VV*r||/||r||=" << coeff_relnorm << std::endl;
		}

		/*
		 * Take a look at the first and last few vectors that for the basis for V and ensure that the columns are orthogonal
		 */

		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_1 = El::LockedView(V,0,0,n,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_2 = El::LockedView(V,0,1,n,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_3 = El::LockedView(V,0,2,n,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_4 = El::LockedView(V,0,3,n,1);

		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_l1 = El::LockedView(V,0,r-1,n,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_l2 = El::LockedView(V,0,r-2,n,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_l3 = El::LockedView(V,0,r-3,n,1);
		const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_l4 = El::LockedView(V,0,r-4,n,1);
		/*
			 elemental2tree(V_1,&temp);
			 temp.Write2File((SAVE_DIR_STR+"v1").c_str(),VTK_ORDER);
			 elemental2tree(V_2,&temp);
			 temp.Write2File((SAVE_DIR_STR+"v2").c_str(),VTK_ORDER);
			 elemental2tree(V_3,&temp);
			 temp.Write2File((SAVE_DIR_STR+"v3").c_str(),VTK_ORDER);
			 elemental2tree(V_4,&temp);
			 temp.Write2File((SAVE_DIR_STR+"v4").c_str(),VTK_ORDER);
			 elemental2tree(V_l1,&temp);
			 temp.Write2File((SAVE_DIR_STR+"vl1").c_str(),VTK_ORDER);
			 elemental2tree(V_l2,&temp);
			 temp.Write2File((SAVE_DIR_STR+"vl2").c_str(),VTK_ORDER);
			 elemental2tree(V_l3,&temp);
			 temp.Write2File((SAVE_DIR_STR+"vl3").c_str(),VTK_ORDER);
			 elemental2tree(V_l4,&temp);
			 temp.Write2File((SAVE_DIR_STR+"vl4").c_str(),VTK_ORDER);
			 */


		/*
		// test that the vectors are orhtogonal
		elemental2tree(V_1,&temp);
		elemental2tree(V_2,&temp2);
		temp.ConjMultiply(&temp2,1);
		std::vector<double> integral = temp.Integrate();
		if(!rank) std::cout << "<V_1, V_2> = " << integral[0] << "+i" << integral[1] << std::endl;

		elemental2tree(V_1,&temp);
		elemental2tree(V_3,&temp2);
		temp.ConjMultiply(&temp2,1);
		integral = temp.Integrate();
		if(!rank) std::cout << "<V_1, V_3> = " << integral[0] << "+i" << integral[1] << std::endl;

		elemental2tree(V_1,&temp);
		elemental2tree(V_l1,&temp2);
		temp.ConjMultiply(&temp2,1);
		integral = temp.Integrate();
		if(!rank) std::cout << "<V_1, V_end> = " << integral[0] << "+i" << integral[1] << std::endl;
		*/

		/*
		 * Test the orthogonality of the matrices via elemental
		 */

		{
			El::DistMatrix<El::Complex<double>,VR,STAR> I(g);
			El::DistMatrix<El::Complex<double>,VR,STAR> UtU(g);
			El::Zeros(UtU,r,r);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,U,U,beta,UtU);
			El::Identity(I,r,r);
			El::Axpy(-1.0,I,UtU);
			double ortho_diff = El::TwoNorm(UtU)/El::TwoNorm(I);
			if(!rank) std::cout << "Ortho diff U: " << ortho_diff << std::endl;

			El::DistMatrix<El::Complex<double>,VR,STAR> VtV(g);
			El::Zeros(VtV,r,r);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V,V,beta,VtV);
			El::Axpy(-1.0,I,VtV);
			ortho_diff = El::TwoNorm(VtV)/El::TwoNorm(I);
			if(!rank) std::cout << "Ortho diff V: " << ortho_diff << std::endl;
		}

		/*
		 * Take a look at the spectrum
		 */

		//El::Display(S,"Sigma_G");

		/*
		 * s_1*||V*w||
		 */

		{
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> rand(g);
			El::Gaussian(rand,n,1);
			double sig_1 = El::RealPart(S.Get(0,0));
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vtr(g);
			El::Zeros(Vtr,r,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V,rand,beta,Vtr);
			double Vtr_norm = El::TwoNorm(Vtr);

			if(!rank) std::cout << "s_1 * ||V*r||=" << sig_1*Vtr_norm << std::endl;
		}

		/*
		 * ||G_v1 - s_1 u_1||
		 */
		{
				const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_1 = El::LockedView(V,0,0,n,1);
				double sig_1 = El::RealPart(S.Get(0,0));
				El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gv1(g);
				El::Zeros(Gv1,m,1);
				A(V_1,Gv1);

				const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_1 = El::LockedView(U,0,0,m,1);
				El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp_vec(g);
				El::Zeros(temp_vec,m,1);
				El::Axpy(sig_1,U_1,temp_vec);
				El::Axpy(-1.0,Gv1,temp_vec);
				double norm_diff = El::TwoNorm(temp_vec)/sig_1;

				if(!rank) std::cout << "||Av_1 - s_1 u_1||/s_1=" << norm_diff << std::endl;
		}

		/*
		 * G*u_1 - s_1 v_1||
		 */
		{
				const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_1 = El::LockedView(V,0,0,n,1);
				double sig_1 = El::RealPart(S.Get(0,0));
				const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_1 = El::LockedView(U,0,0,m,1);
				El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp_vec(g);
				El::Zeros(temp_vec,n,1);
				El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gtu1(g);
				El::Zeros(Gtu1,n,1);
				At(U_1,Gtu1);

				El::Axpy(sig_1,V_1,temp_vec);
				El::Axpy(-1.0,Gtu1,temp_vec);
				double norm_diff = El::TwoNorm(temp_vec)/sig_1;

				if(!rank) std::cout << "||A*u_1 - s_1 v_1||/s_1=" << norm_diff << std::endl;
		}

		/*
		 * ensure that ||G eta|| <=s1*||V'eta||
		 */
		{
			double sig_1 = El::RealPart(S.Get(0,0));
			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> rand(g);
			El::Gaussian(rand,n,1);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gr2(g);
			El::Zeros(Gr2,m,1);
			A(rand,Gr2);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vt_Gr2(g);
			El::Zeros(Vt_Gr2,r,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V,rand,beta,Vt_Gr2);

			double g_r_norm2 = El::TwoNorm(Gr2);
			double Vt_Gr_norm = El::TwoNorm(Vt_Gr2);
			if(!rank) std::cout << "||Aw|| <= s_1*||A*w||=" << (g_r_norm2 <= sig_1*Vt_Gr_norm) << std::endl;
		}
	}

return 0;
}


int faims_test(MPI_Comm &comm, int N_d_sugg, int R_d, int N_s_sugg, int R_s, int R_b, int n_scatters, double gamma){
	//int R_d = 25;
	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel_1000;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj_1000;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	double rsvd_tol = 0.001;

	//int data_dof = 2;

	// Set grid
	El::Grid g(comm);

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////
	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;


	std::vector<double> d_locs = fib_sph(N_d_sugg, 0.45, comm);
	int lN_d = d_locs.size()/3;
	int N_d;
	MPI_Allreduce(&lN_d, &N_d, 1, MPI_INT, MPI_SUM, comm);
	if(!rank) std::cout << "Number of detectors generated=" << N_d << std::endl;

	//std::vector<double> ev_coord = unif_point_distrib(8000,0.25,0.75,comm);
	std::vector<double> scatter_coord = unif_line(n_scatters, 0, 0.25, 0.75, comm);
	int lN_scatter = scatter_coord.size()/3;
	int N_scatter;
	MPI_Allreduce(&lN_scatter, &N_scatter, 1, MPI_INT, MPI_SUM, comm);
	if(!rank) std::cout << "Number of scatterers generated=" << N_scatter << std::endl;

	std::vector<double> src_coord = fib_sph(N_s_sugg, 0.4, comm);
	int lN_s = src_coord.size()/3;
	int N_s;
	MPI_Allreduce(&lN_s, &N_s, 1, MPI_INT, MPI_SUM, comm);
	if(!rank) std::cout << "Number of sources generated=" << N_s << std::endl;

	//////////////////////////////////////////////////////////////
	// Some data setup
	//////////////////////////////////////////////////////////////

	// initialize the trees
	ParticleTree temp(scatter_coord,d_locs,scatter_coord,bndry,comm);
	ParticleTree eta(scatter_coord,d_locs,scatter_coord,bndry,comm);
	eta.ApplyFn(eta_fn,sv);

	// Tree sizes
	int m = temp.m;
	int M = temp.M; //m is small, the number of detectors
	int n = temp.n;
	int N = temp.N; // n is the discretization of the whole space
	int N_disc = N;

	// Set the scalars for multiplying in Gemm
	auto alpha = El::Complex<double>(1.0);
	auto beta = El::Complex<double>(0.0);

	//////////////////////////////////////////////////////////////
	// File name prefix
	//////////////////////////////////////////////////////////////
	std::string params = "-" + std::to_string((long long)(N_d)) + "-" + std::to_string((long long)(R_d)) + "-" + std::to_string((long long)(N_s)) + "-" + std::to_string((long long)(R_s)) + "-" + std::to_string((long long)(R_b));
	if(!rank) std::cout << "Params " << params << std::endl;


	El::DistMatrix<double,El::VC,El::STAR> scatter_coord_el(g);
	El::Zeros(scatter_coord_el,N_scatter*3,1);
	coordvec2elemental(scatter_coord,scatter_coord_el);
	El::Write(scatter_coord_el,SAVE_DIR_STR+"scatter_coord"+params,El::ASCII);


	
	//////////////////////////////////////////////////////////////
	// Compute the incident field
	//////////////////////////////////////////////////////////////
	
	//El::DistMatrix<El::Complex<double>,El::VC,El::STAR> inc_field_el(g);
	//El::Zeros(inc_field_el,N,1);

	//ParticleTree inc_field(src_coord,src_value,scatter_coord,scatter_coord,bndry,comm);
	//inc_field.SetupFMM(kernel);
	//inc_field.RunFMM();
	//tree2elemental(inc_field,inc_field_el,tv);
	//El::Display(inc_field_el,"inc_field");

	//El::DistMatrix<El::Complex<double>,El::VC,El::STAR> inc_field_el_c = inc_field_el;


	/////////////////////////////////////////////////////////////////
	// Low rank factorization of U
	/////////////////////////////////////////////////////////////////
	
	FMM_op U(src_coord,scatter_coord, kernel, bndry,"U",comm);
	FMM_op Ut(scatter_coord,src_coord, kernel_conj, bndry,"U*",comm);
	if(!rank) std::cout << "Low Rank Factorization of U" << std::endl;

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_U(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR>	S_U(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_U(g);

	rsvd::RSVDCtrl ctrl_U;
	ctrl_U.m=N_disc;
	ctrl_U.n=N_s;
	ctrl_U.r=R_s;
	ctrl_U.l=10;
	ctrl_U.q=0;
	ctrl_U.tol=rsvd_tol;
	ctrl_U.max_sz=N_s;
	ctrl_U.adap=rsvd::FIXED;
	ctrl_U.orientation=rsvd::NORMAL;
	rsvd::rsvd(U_U,S_U,V_U,U,Ut,ctrl_U);
	R_s = ctrl_U.r;
	if(!rank) std::cout << "R_s = " << R_s << std::endl;

	svd_test(U,Ut,U_U,S_U,V_U,N_disc,N_s,R_s,g,comm);


	//////////////////////////////////////////////////////////////
	// Compute the forward scattering
	//////////////////////////////////////////////////////////////
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> eta_el(g);
	El::Zeros(eta_el,N,1);
	tree2elemental(eta,eta_el,sv);
	El::Write(eta_el,SAVE_DIR_STR+"eta_actual"+params,El::ASCII);
	
	El::DiagonalScale(El::RIGHT,El::NORMAL,S_U,U_U);
	auto U_hat = U_U;
	auto U_hat_eta = U_U;
	El::DiagonalScale(El::LEFT,El::NORMAL,eta_el,U_hat_eta);

	FMM_op G(scatter_coord,d_locs, kernel, bndry,"G",comm);
	FMM_op Gt(d_locs,scatter_coord, kernel_conj, bndry,"G*",comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> phi_hat(g);
	El::Zeros(phi_hat,N_d,R_s);

	for(int i=0;i<R_s;i++){
		const auto u_hat_eta_i = El::LockedView(U_hat_eta,0,i,N_disc,1);
		auto phi_hat_i = El::View(phi_hat,0,i,N_d,1);

		G(u_hat_eta_i, phi_hat_i);
	}

	/////////////////////////////////////////////////////////////////
	// Low rank factorization of G
	/////////////////////////////////////////////////////////////////
	if(!rank) std::cout << "Low Rank Factorization of G" << std::endl;

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
	ctrl_G.adap=rsvd::FIXED;
	ctrl_G.orientation=rsvd::NORMAL;
	rsvd::rsvd(U_G,S_G,V_G,G,Gt,ctrl_G);
	R_d = ctrl_G.r;
	if(!rank) std::cout << "R_d = " << R_d << std::endl;

	svd_test(G,Gt,U_G,S_G,V_G,N_d,N_disc,R_d,g,comm);


	////////////////////////////////////////////////////////////////////
	// Compute D = U_G*Phi_hat
	////////////////////////////////////////////////////////////////////
	
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> D(g);
	El::Zeros(D,R_d,R_s);
	El::Gemm(El::ADJOINT,El::NORMAL,alpha,U_G,phi_hat,beta,D);

	////////////////////////////////////////////////////////////////////
	// Compute Randomized SVD of B
	////////////////////////////////////////////////////////////////////

	B_op   B(S_G,V_G,U_hat,"B",g,comm);
	Bt_op Bt(S_G,V_G,U_hat,"B",g,comm);
	{

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> x(g);
		El::Gaussian(x,N_disc,1);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Bx(g);
		El::Zeros(Bx,R_d*R_s,1);

		B(x,Bx);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> y(g);
		El::Gaussian(y,R_d*R_s,1);

		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Bty(g);
		El::Zeros(Bty,N_disc,1);

		Bt(y,Bty);

		El::Complex<double> Bxy = El::Dot(Bx,y);
		El::Complex<double> xBty = El::Dot(x,Bty);
		if(!rank) std::cout << "Bxy=" << Bxy << std::endl;
		if(!rank) std::cout << "xBty=" << xBty << std::endl;
		double diff = El::Abs(Bxy - xBty)/El::Abs(Bxy);

		std::string name = "inner prod test";
		test_less(1e-6,diff,name,comm);


	}
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_B(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR>	S_B(g);
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_B(g);

	rsvd::RSVDCtrl ctrl_B;
	ctrl_B.m=R_d*R_s;
	ctrl_B.n=N_disc;
	ctrl_B.r=R_b;
	ctrl_B.l=10;
	ctrl_B.q=0;
	ctrl_B.tol=rsvd_tol;
	ctrl_B.max_sz=R_d*R_s;
	ctrl_B.adap=rsvd::FIXED;
	ctrl_B.orientation=rsvd::NORMAL;
	rsvd::rsvd(U_B,S_B,V_B,B,Bt,ctrl_B);
	R_b = ctrl_B.r;
	if(!rank) std::cout << "R_b = " << R_b << std::endl;
	svd_test(B,Bt,U_B,S_B,V_B,R_d*R_s,N_disc,R_b,g,comm);

	////////////////////////////////////////////////////////////////////
	// Let's try inverting now
	////////////////////////////////////////////////////////////////////
	
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> d_vec(g);
	El::Zeros(d_vec,R_d*R_s,1);
	for(int i=0;i<R_s;i++){
		const auto d_i = El::LockedView(D,0,i,R_d,1);
		std::vector<int> rows(R_d);
		std::iota(rows.begin(), rows.end(),i*R_d);
		std::vector<int> cols = {0};
		El::SetSubmatrix(d_vec,rows,cols,d_i);
	}

	{ // lets test this B thing
		El::DistMatrix<El::Complex<double>,El::VC,El::STAR> d_vec2(g);
		El::Zeros(d_vec2,R_d*R_s,1);
		B(eta_el,d_vec2);
		El::Axpy(El::Complex<double>(-1.0),d_vec,d_vec2);
		std::cout << "Is B working ok???" << El::TwoNorm(d_vec2)/El::TwoNorm(d_vec) << std::endl;
	}

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Ut_Bd(g);
	El::Zeros(Ut_Bd,R_b,1);
	El::Gemv(El::ADJOINT,alpha,U_B,d_vec,beta,Ut_Bd);


	// apply some regularization
	auto sigmaMap = 
		[&gamma]( El::Complex<double> sigma ) 
		{ return sigma / (sigma*sigma + gamma*gamma); };
	EntrywiseMap( S_B, function<El::Complex<double>(El::Complex<double>)>(sigmaMap) );

	El::DiagonalScale(El::LEFT,El::NORMAL,S_B,Ut_Bd);

	auto S_invUt_Bd = Ut_Bd;

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> eta_recon(g);
	El::Zeros(eta_recon,N_disc,1);
	El::Gemv(El::NORMAL,alpha,V_B,S_invUt_Bd,beta,eta_recon);

	El::Write(eta_recon,SAVE_DIR_STR+"eta_recon"+params,El::ASCII);

	El::Axpy(El::Complex<double>(-1.0),eta_el,eta_recon);
	double norm_err = El::TwoNorm(eta_recon)/El::TwoNorm(eta_el);
	if(!rank) std::cout << "||eta - eta_recon||/||eta||=" << norm_err << std::endl;

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
	isotree::ParticleTree::max_pts = 10000;
	isotree::ParticleTree::mult_order = El::Input("-fmm_m","Multipole order",10);
	isotree::ParticleTree::maxdepth = El::Input("-max_depth","Maximum tree depth",3);
	isotree::ParticleTree::adap = El::Input("-adap","Adaptivity for tree construction",true);
	int N_d = El::Input("-N_d","Number of detectors",100);
	int R_d = El::Input("-R_d","Reduced detector rank",25);
	int N_s = El::Input("-N_s","Number of sources",100);
	int R_s = El::Input("-R_s","Reduced source rank",25);
	int R_b = El::Input("-R_b","Reduced B rank",25);
	int n_scatters = El::Input("-N_scatter","Number of scatterers",25);
	double gamma = El::Input("-gamma","Regularization parameter",0.001);
	SAVE_DIR_STR = El::Input("-dir","Directory for saving the functions and the matrices to",".");
	if(!rank) std::cout << SAVE_DIR_STR << std::endl;
	El::PrintInputReport();

	///////////////////////////////////////////////
	// TESTS
	//////////////////////////////////////////////
	faims_test(comm,N_d,R_d,N_s,R_s,R_b,n_scatters,gamma);

	El::Finalize();

	return 0;
}
