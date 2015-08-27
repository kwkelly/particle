
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


int faims_test(MPI_Comm &comm, int N_d_sugg, int R_d, int N_s_sugg, int R_s, int R_b, int n_scatters){
	//int R_d = 25;
	// Set up some trees
	const pvfmm::Kernel<double>* kernel=&helm_kernel_10;
	const pvfmm::Kernel<double>* kernel_conj=&helm_kernel_conj_10;
	pvfmm::BoundaryType bndry=pvfmm::FreeSpace;
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	double rsvd_tol = 0.001;

	int data_dof = 2;

	//////////////////////////////////////////////////////////////
	// Set up the source and detector locations
	//////////////////////////////////////////////////////////////
	ParticleTree::AVec tv = ParticleTree::AVec::trg;
	ParticleTree::AVec sv = ParticleTree::AVec::src;


	//std::vector<double> d_locs = unif_plane(N_d_sugg, 0, 0.9, comm);
	//std::vector<double> d_locs = rand_sph_point_distrib(N_d_sugg, 0.45, comm);
	std::vector<double> d_locs = fib_sph(N_d_sugg, 0.45, comm);
	std::cout << "d_locs" << std::endl;
//	for(int i=0;i<d_locs.size();i+=3){
//		std::cout << "i=" << i << std::endl;
//		std::cout << d_locs[i] << std::endl;
//		std::cout << d_locs[i+1] << std::endl;
//		std::cout << d_locs[i+2] << std::endl;
//		std::cout << sqrt((d_locs[i]-0.5)*(d_locs[i]-0.5) + (d_locs[i+1]-0.5)*(d_locs[i+1]-0.5) + (d_locs[i+2]-0.5)*(d_locs[i+2]-0.5)) << std::endl;
//	}

	int lN_d = d_locs.size()/3;
	int N_d;
	MPI_Allreduce(&lN_d, &N_d, 1, MPI_INT, MPI_SUM, comm);
	if(!rank) std::cout << "Number of detectors generated=" << N_d << std::endl;

	//std::vector<double> ev_coord = unif_point_distrib(8000,0.25,0.75,comm);
	std::vector<double> scatter_coord = unif_line(n_scatters, 0, 0.25, 0.75, comm);
	//std::cout << "scatter_coord" << std::endl;
	//for(int i=0;i<scatter_coord.size();i++){
	//	std::cout << scatter_coord[i] << std::endl;
	//}

	std::vector<double> src_coord;
	std::vector<double> src_value;
	if(!rank) src_coord = {0.5,0.5,0.7};
	if(!rank) src_value = {1.0,0.0};

	//////////////////////////////////////////////////////////////
	// Some data setup
	//////////////////////////////////////////////////////////////

	ParticleTree temp(scatter_coord,d_locs,scatter_coord,bndry,comm);
	ParticleTree mask(scatter_coord,d_locs,scatter_coord,bndry,comm);
	ParticleTree eta(scatter_coord,d_locs,scatter_coord,bndry,comm);
	mask.ApplyFn(mask_fn,sv);
	eta.ApplyFn(eta_fn,sv);

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


	//////////////////////////////////////////////////////////////
	// File name prefix
	//////////////////////////////////////////////////////////////
	std::string params = "-" + std::to_string((long long)(ParticleTree::maxdepth)) + "-" + std::to_string((long long)(N_d)) + "-" + std::to_string((long long)(R_d));
	if(!rank) std::cout << "Params " << params << std::endl;


	//////////////////////////////////////////////////////////////
	// Compute the incident field
	//////////////////////////////////////////////////////////////
	
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> inc_field_el(g);
	El::Zeros(inc_field_el,N,1);

	ParticleTree inc_field(src_coord,src_value,scatter_coord,scatter_coord,bndry,comm);
	inc_field.SetupFMM(kernel);
	inc_field.RunFMM();
	tree2elemental(inc_field,inc_field_el,tv);
	//El::Display(inc_field_el,"inc_field");

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> inc_field_el_c = inc_field_el;

	//////////////////////////////////////////////////////////////
	// Compute the forward scattering
	//////////////////////////////////////////////////////////////

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> eta_el(g);
	El::Zeros(eta_el,N,1);
	tree2elemental(eta,eta_el,sv);
	//El::Display(eta_el,"eta");

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> eta_u(g);
	El::Zeros(eta_u,N,1);
	El::DiagonalScale(El::LEFT, El::NORMAL, eta_el, inc_field_el_c);

	Global_to_det_op G(scatter_coord,d_locs,mask_fn, kernel, bndry,"G",comm);
	Det_to_global_op Gt(d_locs,scatter_coord,mask_fn, kernel_conj, bndry,"G*",comm);

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> scattered_field_el(g);
	El::Zeros(scattered_field_el,N_d,1);

	G(inc_field_el_c, scattered_field_el);
	//El::Display(scattered_field_el,"scattered_field");

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
			//elemental2tree(r,temp,sv);
			//temp.Multiply(mask,sv,sv);
			//tree2elemental(temp,r,sv);
			//El::Display(r,"r");

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Vt_Gr(g);
			El::Zeros(Vt_Gr,R_d,1);
			El::Gemm(El::ADJOINT,El::NORMAL,alpha,V_G,r,beta,Vt_Gr);

			El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VVt_Gr(g);
			El::Zeros(VVt_Gr,N_disc,1);
			El::Gemm(El::NORMAL,El::NORMAL,alpha,V_G,Vt_Gr,beta,VVt_Gr);
			//El::Display(VVt_Gr,"VV*r");

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

		//El::Display(S_G,"Sigma_G");

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
			for(int i = 0; i < 10;i++){
				const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_1 = El::LockedView(V_G,0,i,N_disc,1);
				double sig_1 = El::RealPart(S_G.Get(i,0));
				El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gv1(g);
				El::Zeros(Gv1,N_d,1);
				G(V_G_1,Gv1);

				const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_G_1 = El::LockedView(U_G,0,i,N_d,1);
				El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp_vec(g);
				El::Zeros(temp_vec,N_d,1);
				El::Axpy(sig_1,U_G_1,temp_vec);
				El::Axpy(-1.0,Gv1,temp_vec);
				double norm_diff = El::TwoNorm(temp_vec)/sig_1;

				if(!rank) std::cout << "||Gv_1 - s_1 u_1||/s_1=" << norm_diff << std::endl;
			}
		}

		/*
		 * G*u_1 - s_1 v_1||
		 */
		{
			for(int i = 0; i < 10;i++){
				const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> V_G_1 = El::LockedView(V_G,0,i,N_disc,1);
				double sig_1 = El::RealPart(S_G.Get(i,0));
				const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> U_G_1 = El::LockedView(U_G,0,i,N_d,1);
				El::DistMatrix<El::Complex<double>,El::VC,El::STAR> temp_vec(g);
				El::Zeros(temp_vec,N_disc,1);
				El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Gtu1(g);
				El::Zeros(Gtu1,N_disc,1);
				Gt(U_G_1,Gtu1);

				El::Axpy(sig_1,V_G_1,temp_vec);
				El::Axpy(-1.0,Gtu1,temp_vec);
				double norm_diff = El::TwoNorm(temp_vec)/sig_1;

				if(!rank) std::cout << "||G*u_1 - s_1 v_1||/s_1=" << norm_diff << std::endl;
			}
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


	////////////////////////////////////////////////////////////////////
	// Let's try inverting now
	////////////////////////////////////////////////////////////////////

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> Ut_Gd(g);
	El::Zeros(Ut_Gd,R_d,1);
	El::Gemv(El::ADJOINT,alpha,U_G,scattered_field_el,beta,Ut_Gd);

	El::Display(S_G, "S_G");
	
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> S_invUt_Gd(g);
	El::Zeros(S_invUt_Gd,R_d,1);
	El::DiagonalSolve(El::LEFT,El::NORMAL,S_G,Ut_Gd);

	S_invUt_Gd = Ut_Gd;

	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> VS_invUt_Gd(g);
	El::Zeros(VS_invUt_Gd,N_disc,1);
	El::Gemv(El::NORMAL,alpha,V_G,S_invUt_Gd,beta,VS_invUt_Gd);

	//El::Display(inc_field_el, "incident_field");
	El::DistMatrix<El::Complex<double>,El::VC,El::STAR> u_invVS_invUt_Gd(g);
	El::Zeros(u_invVS_invUt_Gd,N_disc,1);
	El::DiagonalSolve(El::LEFT,El::NORMAL,inc_field_el,VS_invUt_Gd);

	u_invVS_invUt_Gd = VS_invUt_Gd;

	//El::Display(u_invVS_invUt_Gd,"eta_recon");

	El::Axpy(El::Complex<double>(-1.0),eta_el,u_invVS_invUt_Gd);
	double norm_err = El::TwoNorm(u_invVS_invUt_Gd)/El::TwoNorm(eta_el);
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
	isotree::ParticleTree::max_pts = 1000;
	isotree::ParticleTree::mult_order = El::Input("-fmm_m","Multipole order",10);
	isotree::ParticleTree::maxdepth = El::Input("-max_depth","Maximum tree depth",3);
	isotree::ParticleTree::adap = El::Input("-adap","Adaptivity for tree construction",true);
	int N_d = El::Input("-N_d","Number of detectors",100);
	int R_d = El::Input("-R_d","Reduced detector rank",25);
	int N_s = El::Input("-N_s","Number of sources",100);
	int R_s = El::Input("-R_s","Reduced source rank",25);
	int R_b = El::Input("-R_b","Reduced B rank",25);
	int n_scatters = El::Input("-N_scatter","Number of scatterers",25);
	SAVE_DIR_STR = El::Input("-dir","Directory for saving the functions and the matrices to",".");
	if(!rank) std::cout << SAVE_DIR_STR << std::endl;
	El::PrintInputReport();

	///////////////////////////////////////////////
	// TESTS
	//////////////////////////////////////////////
	faims_test(comm,N_d,R_d,N_s,R_s,R_b,n_scatters);

	El::Finalize();

	return 0;
}
