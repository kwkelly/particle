#include "petscsys.h"
#include "invmed_tree.hpp"
#include "pvfmm.hpp"

#pragma once

namespace isotree{

template <class FMM_Mat_t>
int tree2vec(ChebTree<FMM_Mat_t> *tree, Vec& Y);

template <class FMM_Mat_t>
int vec2tree(Vec& Y, ChebTree<FMM_Mat_t> *tree);
}
