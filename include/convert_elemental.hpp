#include "El.hpp"
#include "particle_tree.hpp"
#include <vector>

#pragma once

/*
 * the sum op for a scan operation used in the below function.
 */
template <typename T>
void op(T& v1, const T& v2);

namespace isotree{
/*
 * Convert a pvfmm tree to en elemental vector whose entries are the Chebyshev coefficients
 *
 * This function reorders the data because of the way that elemental stores the data. 
 * Ideally we would have values that are stored consecutively in the pvfmm tree to be stored
 * consecutively in the Elemental vector. Unfortunately, there is no Elemental distribution scheme 
 * which makes this simple. Consequently, the data is reorder in a manner that reduces the amount of 
 * communication required.
 */

template <class FMM_Mat_t, typename T>
int tree2elemental(const ParticleTree &tree, El::DistMatrix<T,El::VC,El::STAR> &Y);

/*
 * Convert an Elemental vector to a pvfmm tree. The opposite of the previous function
 *
 * This function reorders the data because of the way that elemental stores the data. 
 * Ideally we would have values that are stored consecutively in the pvfmm tree to be stored
 * consecutively in the Elemental vector. Unfortunately, there is no Elemental distribution scheme 
 * which makes this simple. Consequently, the data is reorder in a manner that reduces the amount of 
 * communication required.
 */
template <class FMM_Mat_t, typename T>
int elemental2tree(const El::DistMatrix<T,El::VC,El::STAR> &Y, ParticleTree &tree);

/*
 * Convert a std::vector to an elemental vector of the given distribution. As before, thise well reorder the data
 */
int vec2elemental(const std::vector<double> &vec, El::DistMatrix<El::Complex<double>,El::VC,El::STAR > &Y);

/*
 * Convert an elemental vector to a std::vector of the given distribution. As before, thise well reorder the data
 */
int elemental2vec(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y, std::vector<double> &vec);

}
