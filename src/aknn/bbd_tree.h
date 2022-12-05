#ifndef AKNN_BBD_TREE_H
#define AKNN_BBD_TREE_H

#include <vector>
#include <stdint.h>
#include <variant>

#include "vec.h"

enum class NodeType
{
    SPLIT = 0,
    SHRINK,
    LEAF
};

#define NODE_TYPE_BITS 2
#define DIM_BITS 3
#define LEFT_CHILD_BIT 1

#define CUSTOM_DATA_POS (LEFT_CHILD_BIT + DIM_BITS + NODE_TYPE_BITS)
#define LEFT_CHILD_POS (DIM_BITS + NODE_TYPE_BITS)
#define DIM_POS (NODE_TYPE_BITS)

#define NODE_TYPE_MASK ((1 << NODE_TYPE_BITS) - 1)
#define DIM_MASK ((1 << DIM_BITS) - 1)
#define LEFT_CHILD_MASK ((1 << LEFT_CHILD_BIT) - 1)

class Node
{
protected:
    uint64_t customData_nodeType;
public:
    NodeType GetType() const { return static_cast<NodeType>(customData_nodeType & NODE_TYPE_MASK); };
};

// data up to 10^8
// build from left
// size 8 bytes
class SplitNode
{
private:
    // 58 bits right child index, 1 bit left child, 3 bits dimension, 2 bits node type
    uint64_t rightChild_leftChild_splitDim_nodeType;
public:
    SplitNode(int splitDim) {}

    int GetSplitDim() const {
        return (int)((rightChild_leftChild_splitDim_nodeType >> DIM_POS) & DIM_MASK);
    }
    bool HasLeftChild() {

    }
    void SetLeftChild(bool exists) {

    }
    int64_t GetRightChildIndex() const {
        return (int64_t)(rightChild_leftChild_splitDim_nodeType >> CUSTOM_DATA_POS);
    }
    void SetRightChildIndex(int64_t i) const {
        //return (int64_t)(rightChild_leftChild_splitDim_nodeType >> CUSTOM_DATA_POS);
    }
};

// size 8 + 2*sizeof(Vec<FloatT, Dim>) bytes
// float:  8 +  8*Dim
// double: 8 + 16*Dim
template<typename FloatT, int Dim>
class ShrinkNode
{
private:
    uint64_t rightChild_leftChild_nodeType;
    AABB<FloatT, Dim> shrinkBox;
public:
    int64_t GetRightChildIndex() const { return (int64_t)(rightChild_splitDim_nodeType >> CUSTOM_DATA_BIT_POS); }
};

// size 16 bytes
template<typename FloatT, int Dim>
class LeafNode
{
private:
    uint64_t pointsFrom_nodeType;
    uint64_t pointsTo;

public:

    int64_t GetPointsFromIndex() const { return (int64_t)(rightChild_splitDim_nodeType >> NODE_TYPE_BITS); }
    int64_t GetPointsToIndex() const { return (int64_t)(pointsTo); }
};

template<typename FloatT, int Dim>
using NodeVar = std::variant<SplitNode*, ShrinkNode<FloatT, Dim>*, LeafNode<FloatT, Dim>*>;

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

    void AddSplitNode();
};

#endif // AKNN_BBD_TREE_H
