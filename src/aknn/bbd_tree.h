#ifndef AKNN_BBD_TREE_H
#define AKNN_BBD_TREE_H

#include <vector>
#include <stdint.h>

#include "vec.h"

enum class NodeType
{
    SPLIT = 0,
    SHRINK,
    LEAF
};

#define NODE_TYPE_BITS 2
#define DIM_BITS 2

#define CUSTOM_DATA_BIT_POS (DIM_BITS + NODE_TYPE_BITS)
#define NODE_TYPE_MASK ((1 << NODE_TYPE_BITS) - 1)
#define DIM_MASK ((1 << DIM_BITS) - 1)

struct Node
{
    uint64_t customData_nodeType;
};

// data up to 10^8
// build from left
// size 8 bytes
struct SplitNode
{
    // 60 bits right child index, 2 bits dimension, 2 bits node type
    uint64_t rightChild_splitDim_nodeType;
};

// size 8 + 2*sizeof(Vec<FloatT, Dim>) bytes
// float:  8 +  8*Dim
// double: 8 + 16*Dim
template<typename FloatT, int Dim>
struct ShrinkNode
{
    uint64_t rightChild_nodeType;
    AABB<FloatT, Dim> shrinkBox;
};

// size 16 bytes
template<typename FloatT, int Dim>
struct LeafNode
{
    uint64_t pointsFrom_nodeType;
    uint64_t pointsTo;
};

struct KDTree
{

};

template<typename FloatT, int Dim>
struct BBDTree
{
    std::vector<Node> nodes;
    std::vector<Vec<FloatT, Dim>> points;
    AABB<FloatT, Dim> pointsBox;

    Node* GetRoot();
};

#endif // AKNN_BBD_TREE_H
