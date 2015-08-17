#include "El.hpp"
#include "particle_tree.hpp"
#include <vector>

#ifndef CONVERT_ELEMENTAL_HPP
#define CONVERT_ELEMENTAL_HPP

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

template <typename T>
void tree2elemental(const ParticleTree &tree, El::DistMatrix<T,El::VC,El::STAR> &Y, ParticleTree::AVec av);

/*
 * Convert an Elemental vector to a pvfmm tree. The opposite of the previous function
 *
 * This function reorders the data because of the way that elemental stores the data. 
 * Ideally we would have values that are stored consecutively in the pvfmm tree to be stored
 * consecutively in the Elemental vector. Unfortunately, there is no Elemental distribution scheme 
 * which makes this simple. Consequently, the data is reorder in a manner that reduces the amount of 
 * communication required.
 */
template <typename T>
void elemental2tree(const El::DistMatrix<T,El::VC,El::STAR> &Y, ParticleTree &tree, ParticleTree::AVec av);

/*
 * Convert a std::vector to an elemental vector of the given distribution. As before, thise well reorder the data
 */
void vec2elemental(const std::vector<double> &vec, El::DistMatrix<El::Complex<double>,El::VC,El::STAR > &Y);

/*
 * Convert an elemental vector to a std::vector of the given distribution. As before, thise well reorder the data
 */
void elemental2vec(const El::DistMatrix<El::Complex<double>,El::VC,El::STAR> &Y, std::vector<double> &vec);

}

#endif
