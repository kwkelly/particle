#include "El.hpp"
#include <vector>
#include <omp.h>
#include "particle_tree.hpp"
#include "par_scan/gen_scan.hpp"
#include "convert_elemental.hpp"

/*
 * the sum op for a scan operation used in the below function.
 */
template <typename T>
void op(T& v1, const T& v2){
	v1+=v2;
}

namespace isotree {

template<typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &ob)
{
	for(int i=0;i<ob.size();i++){
	  stream << ob[i] << '\n';
	}
	return stream;
}

template
std::ostream &operator<<(std::ostream &stream, const std::vector<int> &ob);

template
std::ostream &operator<<(std::ostream &stream, const std::vector<double> &ob);

template
std::ostream &operator<<(std::ostream &stream, const std::vector<El::Complex<double>> &ob);


std::vector<int> exscan(const std::vector<int> &x){
	int ll = x.size();
	std::vector<int> y(ll);
	y[0] = 0;

	for(int i=1;i<ll;i++){
		y[i] = y[i-1] + x[i-1];
	}
	
	return y;
}


template <typename T>
void elemental2tree(const El::DistMatrix<T,El::VC,El::STAR> &Y, ParticleTree &tree, ParticleTree::AVec av){
	if(av == ParticleTree::AVec::trg) elemental2vec(Y,tree.trg_value);
	if(av == ParticleTree::AVec::src) elemental2vec(Y,tree.src_value);
	
	return;
}


template <typename T>
void tree2elemental(const ParticleTree &tree, El::DistMatrix<T,El::VC,El::STAR> &Y, ParticleTree::AVec av){
	if(av == ParticleTree::AVec::trg) vec2elemental(tree.trg_value,Y);
	if(av == ParticleTree::AVec::src) vec2elemental(tree.src_value,Y);

	return;
}

void vec2elemental(const std::vector<double> &vec, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y){

	int data_dof=2;

	int nlocal,gsize; //local elements, start p_id, global size
	//double *pt_array; // will hold local array
	int r,q,rq; //Grid sizes
	int nbigs; //Number of large sends (i.e. send 1 extra data point)
	int pstart; // p_id of nstart
	int rank = El::mpi::WorldRank(); //p_id
	int send_size; // base send size
	bool print = rank == -1; 


	// Get Grid and associated params
	const El::Grid* g = &(Y.Grid());
	r = g->Height();
	q = g->Width();
	MPI_Comm comm = (g->Comm()).comm;

	// Get sizes, array in petsc 
	nlocal = vec.size()/data_dof;
	int nstart = 0;
	MPI_Exscan(&nlocal,&nstart,1,MPI_INT,MPI_SUM,comm);
	//VecGetOwnershipRange(pt_vec,&nstart,NULL);

	//Find processor that nstart belongs to, number of larger sends
	rq = r * q;
	pstart = nstart % rq; //int div
	nbigs = nlocal % rq;
	send_size = nlocal/rq;
	
	if(print){
		std::cout << "r: " << r << " q: " << q <<std::endl;
		std::cout << "nstart: " << nstart << std::endl;
		std::cout << "ps: " << pstart << std::endl;
		std::cout << "nbigs: " << nbigs << std::endl;
		std::cout << "send_size: " << send_size << std::endl;
	}

	// Make send_lengths
	std::vector<int> send_lengths(rq);
	std::fill(send_lengths.begin(),send_lengths.end(),send_size);
	if(nbigs >0){
		for(int j=0;j<nbigs;j++){
			send_lengths[(pstart + j) % rq] += 1;
		}
	}

	// Make send_disps
	std::vector<int> send_disps = exscan(send_lengths);

	std::vector<El::Complex<double>> indata(nlocal);
	// copy the data from an ffm tree to into a local vec of complex data for sending #pragma omp parallel for
	El::Complex<double> val;
	for(int i=0;i<nlocal;i++){
		El::SetRealPart(val,vec[2*i+0]);
		El::SetImagPart(val,vec[2*i+1]);
		indata[i] = val;
	}


	// Make send_dataA, i.e. reorder the data
	std::vector<El::Complex<double>> send_data(nlocal);
	for(int proc=0;proc<rq;proc++){
		int offset = send_disps[proc];
		int base_idx = (proc - pstart + rq) % rq; 
		for(int j=0; j<send_lengths[proc]; j++){
			int idx = base_idx + (j * rq);
			send_data[offset + j] = indata[idx];
		}
	}

	// Do all2all to get recv_lengths
	std::vector<int> recv_lengths(rq);
	MPI_Alltoall(&send_lengths[0], 1, MPI_INT, &recv_lengths[0], 1, MPI_INT,comm);

	// Scan to get recv_disps
	std::vector<int> recv_disps = exscan(recv_lengths);

	// Do all2allv to get data on correct processor
	El::Complex<double> * recv_data = Y.Buffer();
	//MPI_Alltoallv(&send_data[0],&send_lengths[0],&send_disps[0],MPI_DOUBLE, \
	//		&recv_data[0],&recv_lengths[0],&recv_disps[0],MPI_DOUBLE,comm);
	El::mpi::AllToAll(&send_data[0], &send_lengths[0], &send_disps[0], recv_data,&recv_lengths[0],&recv_disps[0],comm);

	if(print){
		std::cout << "Send data: " <<std::endl << send_data <<std::endl;
		std::cout << "Send lengths: " <<std::endl << send_lengths <<std::endl;
		std::cout << "Send disps: " <<std::endl << send_disps <<std::endl;
		std::cout << "Recv data: " <<std::endl << recv_data <<std::endl;
		std::cout << "Recv lengths: " <<std::endl << recv_lengths <<std::endl;
		std::cout << "Recv disps: " <<std::endl << recv_disps <<std::endl;
	}

	return;
}




void elemental2vec(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y, std::vector<double> &vec){
	
	assert((Y.DistData().colDist == El::STAR) and (Y.DistData().rowDist == El::VC));

	int data_dof=2;

	//double *pt_array,*pt_perm_array;
	int r,q,ll,rq; // el vec info
	int nbigs; //Number of large recv (i.e. recv 1 extra data point)
	int pstart; // p_id of nstart
	int rank = El::mpi::WorldRank(); //p_id
	int recv_size; // base recv size
	bool print = (rank == -1); 

	// Get el vec info
	ll = Y.Height();
	const El::Grid* g = &(Y.Grid());
	r = g->Height();
	q = g->Width();
	MPI_Comm comm = (g->Comm()).comm;

	int omp_p=omp_get_max_threads();
	
	// Get petsc vec params
	//VecGetLocalSize(pt_vec,&nlocal);
	int nlocal = (vec.size())/data_dof;
	if(print) std::cout << "m: " << std::endl;
	int nstart = 0;
	//VecGetArray(pt_vec,&pt_array);
	//VecGetOwnershipRange(pt_vec,&nstart,NULL);
	MPI_Exscan(&nlocal,&nstart,1,MPI_INT,MPI_SUM,comm);

	// Determine who owns the first element we want
	rq = r * q;
	pstart = nstart % rq;
	nbigs = nlocal % rq;
	recv_size = nlocal / rq;
	
	if(print){
		std::cout << "r: " << r << " q: " << q <<std::endl;
		std::cout << "nstart: " << nstart << std::endl;
		std::cout << "ps: " << pstart << std::endl;
		std::cout << "nbigs: " << nbigs << std::endl;
		std::cout << "recv_size: " << recv_size << std::endl;
	}

	// Make recv sizes
	std::vector<int> recv_lengths(rq);
	std::fill(recv_lengths.begin(),recv_lengths.end(),recv_size);
	if(nbigs >0){
		for(int i=0;i<nbigs;i++){
			recv_lengths[(pstart + i) % rq] += 1;
		}
	}

	// Make recv disps
	std::vector<int> recv_disps = exscan(recv_lengths);

	// All2all to get send sizes
	std::vector<int> send_lengths(rq);
	MPI_Alltoall(&recv_lengths[0], 1, MPI_INT, &send_lengths[0], 1, MPI_INT,comm);

	// Scan to get send_disps
	std::vector<int> send_disps = exscan(send_lengths);

	// Do all2allv to get data on correct processor
	std::vector<El::Complex<double>> recv_data(nlocal);
	std::vector<El::Complex<double>> recv_data_ordered(nlocal);
	//MPI_Alltoallv(el_vec.Buffer(),&send_lengths[0],&send_disps[0],MPI_DOUBLE, \
			&recv_data[0],&recv_lengths[0],&recv_disps[0],MPI_DOUBLE,comm);
	El::mpi::AllToAll(Y.LockedBuffer(), &send_lengths[0], &send_disps[0], &recv_data[0],&recv_lengths[0],&recv_disps[0],comm);
	
	if(print){
		//std::cout << "Send data: " <<std::endl << *el_vec.Buffer() <<std::endl;
		std::cout << "Send lengths: " <<std::endl << send_lengths <<std::endl;
		std::cout << "Send disps: " <<std::endl << send_disps <<std::endl;
		std::cout << "Recv data: " <<std::endl << recv_data <<std::endl;
		std::cout << "Recv lengths: " <<std::endl << recv_lengths <<std::endl;
		std::cout << "Recv disps: " <<std::endl << recv_disps <<std::endl;
	}
	
	// Reorder the data so taht it is in the right order for the fmm tree
	for(int p=0;p<rq;p++){
		int base_idx = (p - pstart + rq) % rq;
		int offset = recv_disps[p];
		for(int i=0;i<recv_lengths[p];i++){
			recv_data_ordered[base_idx + rq*i] = recv_data[offset + i];
		}
	}

	// loop through and put the data into the vector
	#pragma omp parallel for
	for(int i=0;i<nlocal; i++){
		vec[2*i] = El::RealPart(recv_data_ordered[i]);
		vec[2*i+1] = El::ImagPart(recv_data_ordered[i]);
	}

	if(print){std::cout <<"here?"<<std::endl;}

	return;

}


void coordvec2elemental(const std::vector<double> &vec, El::DistMatrix<double,El::VC,El::STAR> &Y){

	int nlocal,gsize; //local elements, start p_id, global size
	//double *pt_array; // will hold local array
	int r,q,rq; //Grid sizes
	int nbigs; //Number of large sends (i.e. send 1 extra data point)
	int pstart; // p_id of nstart
	int rank = El::mpi::WorldRank(); //p_id
	int send_size; // base send size
	bool print = rank == -1; 


	// Get Grid and associated params
	const El::Grid* g = &(Y.Grid());
	r = g->Height();
	q = g->Width();
	MPI_Comm comm = (g->Comm()).comm;

	// Get sizes, array in petsc 
	nlocal = vec.size();
	int nstart = 0;
	MPI_Exscan(&nlocal,&nstart,1,MPI_INT,MPI_SUM,comm);
	//VecGetOwnershipRange(pt_vec,&nstart,NULL);

	//Find processor that nstart belongs to, number of larger sends
	rq = r * q;
	pstart = nstart % rq; //int div
	nbigs = nlocal % rq;
	send_size = nlocal/rq;
	
	if(print){
		std::cout << "r: " << r << " q: " << q <<std::endl;
		std::cout << "nstart: " << nstart << std::endl;
		std::cout << "ps: " << pstart << std::endl;
		std::cout << "nbigs: " << nbigs << std::endl;
		std::cout << "send_size: " << send_size << std::endl;
	}

	// Make send_lengths
	std::vector<int> send_lengths(rq);
	std::fill(send_lengths.begin(),send_lengths.end(),send_size);
	if(nbigs >0){
		for(int j=0;j<nbigs;j++){
			send_lengths[(pstart + j) % rq] += 1;
		}
	}

	// Make send_disps
	std::vector<int> send_disps = exscan(send_lengths);


	// Make send_dataA, i.e. reorder the data
	std::vector<double> send_data(nlocal);
	for(int proc=0;proc<rq;proc++){
		int offset = send_disps[proc];
		int base_idx = (proc - pstart + rq) % rq; 
		for(int j=0; j<send_lengths[proc]; j++){
			int idx = base_idx + (j * rq);
			send_data[offset + j] = vec[idx];
		}
	}

	// Do all2all to get recv_lengths
	std::vector<int> recv_lengths(rq);
	MPI_Alltoall(&send_lengths[0], 1, MPI_INT, &recv_lengths[0], 1, MPI_INT,comm);

	// Scan to get recv_disps
	std::vector<int> recv_disps = exscan(recv_lengths);

	// Do all2allv to get data on correct processor
	double * recv_data = Y.Buffer();
	//MPI_Alltoallv(&send_data[0],&send_lengths[0],&send_disps[0],MPI_DOUBLE, \
	//		&recv_data[0],&recv_lengths[0],&recv_disps[0],MPI_DOUBLE,comm);
	El::mpi::AllToAll(&send_data[0], &send_lengths[0], &send_disps[0], recv_data,&recv_lengths[0],&recv_disps[0],comm);

	if(print){
		std::cout << "Send data: " <<std::endl << send_data <<std::endl;
		std::cout << "Send lengths: " <<std::endl << send_lengths <<std::endl;
		std::cout << "Send disps: " <<std::endl << send_disps <<std::endl;
		std::cout << "Recv data: " <<std::endl << recv_data <<std::endl;
		std::cout << "Recv lengths: " <<std::endl << recv_lengths <<std::endl;
		std::cout << "Recv disps: " <<std::endl << recv_disps <<std::endl;
	}

	return;
}


void elemental2coordvec(const El::DistMatrix<double,El::VC,El::STAR> &Y, std::vector<double> &vec){
	
	assert((Y.DistData().colDist == El::STAR) and (Y.DistData().rowDist == El::VC));

	//double *pt_array,*pt_perm_array;
	int r,q,ll,rq; // el vec info
	int nbigs; //Number of large recv (i.e. recv 1 extra data point)
	int pstart; // p_id of nstart
	int rank = El::mpi::WorldRank(); //p_id
	int recv_size; // base recv size
	bool print = (rank == -1); 

	// Get el vec info
	ll = Y.Height();
	const El::Grid* g = &(Y.Grid());
	r = g->Height();
	q = g->Width();
	MPI_Comm comm = (g->Comm()).comm;

	int omp_p=omp_get_max_threads();
	
	// Get petsc vec params
	//VecGetLocalSize(pt_vec,&nlocal);
	int nlocal = (vec.size());
	if(print) std::cout << "m: " << std::endl;
	int nstart = 0;
	//VecGetArray(pt_vec,&pt_array);
	//VecGetOwnershipRange(pt_vec,&nstart,NULL);
	MPI_Exscan(&nlocal,&nstart,1,MPI_INT,MPI_SUM,comm);

	// Determine who owns the first element we want
	rq = r * q;
	pstart = nstart % rq;
	nbigs = nlocal % rq;
	recv_size = nlocal / rq;
	
	if(print){
		std::cout << "r: " << r << " q: " << q <<std::endl;
		std::cout << "nstart: " << nstart << std::endl;
		std::cout << "ps: " << pstart << std::endl;
		std::cout << "nbigs: " << nbigs << std::endl;
		std::cout << "recv_size: " << recv_size << std::endl;
	}

	// Make recv sizes
	std::vector<int> recv_lengths(rq);
	std::fill(recv_lengths.begin(),recv_lengths.end(),recv_size);
	if(nbigs >0){
		for(int i=0;i<nbigs;i++){
			recv_lengths[(pstart + i) % rq] += 1;
		}
	}

	// Make recv disps
	std::vector<int> recv_disps = exscan(recv_lengths);

	// All2all to get send sizes
	std::vector<int> send_lengths(rq);
	MPI_Alltoall(&recv_lengths[0], 1, MPI_INT, &send_lengths[0], 1, MPI_INT,comm);

	// Scan to get send_disps
	std::vector<int> send_disps = exscan(send_lengths);

	// Do all2allv to get data on correct processor
	std::vector<double> recv_data(nlocal);
	std::vector<double> recv_data_ordered(nlocal);
	//MPI_Alltoallv(el_vec.Buffer(),&send_lengths[0],&send_disps[0],MPI_DOUBLE, \
			&recv_data[0],&recv_lengths[0],&recv_disps[0],MPI_DOUBLE,comm);
	El::mpi::AllToAll(Y.LockedBuffer(), &send_lengths[0], &send_disps[0], &recv_data[0],&recv_lengths[0],&recv_disps[0],comm);
	
	if(print){
		//std::cout << "Send data: " <<std::endl << *el_vec.Buffer() <<std::endl;
		std::cout << "Send lengths: " <<std::endl << send_lengths <<std::endl;
		std::cout << "Send disps: " <<std::endl << send_disps <<std::endl;
		std::cout << "Recv data: " <<std::endl << recv_data <<std::endl;
		std::cout << "Recv lengths: " <<std::endl << recv_lengths <<std::endl;
		std::cout << "Recv disps: " <<std::endl << recv_disps <<std::endl;
	}
	
	// Reorder the data so taht it is in the right order for the fmm tree
	for(int p=0;p<rq;p++){
		int base_idx = (p - pstart + rq) % rq;
		int offset = recv_disps[p];
		for(int i=0;i<recv_lengths[p];i++){
			recv_data_ordered[base_idx + rq*i] = recv_data[offset + i];
		}
	}

	// loop through and put the data into the vector
	for(int i=0;i<nlocal; i++){
		vec[i] = recv_data_ordered[i];
	}

	if(print){std::cout <<"here?"<<std::endl;}

	return;

}


template
void tree2elemental(const ParticleTree& tree, El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y, ParticleTree::AVec av);

template
void elemental2tree(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y, ParticleTree &tree, ParticleTree::AVec av);


}
