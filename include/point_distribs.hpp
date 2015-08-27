#ifndef POINT_DISTRIBS_HPP
#define POINT_DISTRIBS_HPP
/*
 * Generate equidistributed points in the unit cube between lmin and lmax in each dimension
 */
template <class Real_t>
std::vector<Real_t> unif_point_distrib(size_t N, Real_t lmin, Real_t lmax, MPI_Comm comm);


/*
 * Generate uniformly random points points in the unit cube between lmin and lmax in each dimension
 */
template <class Real_t>
std::vector<Real_t> rand_unif_point_distrib(size_t N, Real_t lmin, Real_t lmax, MPI_Comm comm);

/*
 * Generate unformly random points in the the sphere of radius rad centered at (0.5,0.5,0.5)
 */
template <class Real_t>
std::vector<Real_t> rand_sph_point_distrib(size_t N, Real_t rad, MPI_Comm comm);


/*
 * Generate uniformly spaced points on a plane at distance pos from the given axis
 * plane == 0 = yz plane
 * plane == 1 = xz plane
 * plane == 2 = xy plane
 */
template <class Real_t>
std::vector<Real_t> unif_plane(size_t N, int plane, Real_t pos, MPI_Comm comm);


/*
 * Generate the test points listed below. Note that all points are create on all procs
 */
std::vector<double> test_pts();


/*
 * Generate n_points equidistributed points on the
 * surface of a sphere of radius rad
 * http://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
 * Note that this function is not rwritten to  be parallel friendly, i.e. all points
 * will be created on all procs
 */
template <typename Real_t>
std::vector<Real_t> equisph(size_t n_points, Real_t rad, MPI_Comm comm);
/*
 * Generate equidistant points on a line along one of the coordinate axes
 */
template <class Real_t>
std::vector<Real_t> unif_line(size_t N, int plane, Real_t lmin, Real_t lmax, MPI_Comm comm);

/*
 * Generate n_points equidistributed points on the
 * surface of a sphere of radius rad
 * http://www.openprocessing.org/sketch/41142
 */
template <class Real_t>
std::vector<Real_t> fib_sph(size_t N, Real_t r, MPI_Comm comm);
 


#endif
