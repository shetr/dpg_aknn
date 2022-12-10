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
#define DIM_BITS 2
#define LEFT_CHILD_BITS 1
#define RIGHT_CHILD_BITS (sizeof(uint64_t) - (LEFT_CHILD_BITS + DIM_BITS + NODE_TYPE_BITS))

#define DIM_POS (NODE_TYPE_BITS)
#define LEFT_CHILD_POS (DIM_BITS + NODE_TYPE_BITS)
#define RIGHT_CHILD_POS (LEFT_CHILD_BITS + DIM_BITS + NODE_TYPE_BITS)

#define NODE_TYPE_MASK ((1 << NODE_TYPE_BITS) - 1)

#define LOW_DIM_MASK ((1 << DIM_BITS) - 1)
#define LOW_LEFT_CHILD_MASK ((1 << LEFT_CHILD_BITS) - 1)
#define LOW_RIGHT_CHILD_MASK ((1 << RIGHT_CHILD_BITS) - 1)

#define DIM_MASK (LOW_DIM_MASK << DIM_POS)
#define LEFT_CHILD_MASK (LOW_LEFT_CHILD_MASK << LEFT_CHILD_POS)
#define RIGHT_CHILD_MASK (LOW_RIGHT_CHILD_MASK << RIGHT_CHILD_POS)

uint64_t GetBits(uint64_t storage, uint64_t bitPos, uint64_t lowMask) {
    return (storage >> bitPos) & lowMask;
}

void SetBits(uint64_t& storage, uint64_t bits, uint64_t bitPos, uint64_t mask) {
    storage = (storage & (~mask)) | (bits << bitPos);
}

class Node
{
protected:
    uint64_t _customData_nodeType;
public:
    Node(NodeType nodeType) : _customData_nodeType(static_cast<uint64_t>(nodeType)) {}

    NodeType GetType() const { return static_cast<NodeType>(_customData_nodeType & NODE_TYPE_MASK); };
};

class InnerNode : public Node
{
public:
    InnerNode(NodeType nodeType) : Node(nodeType) {}

    bool HasLeftChild() {
        return (bool)((_customData_nodeType >> LEFT_CHILD_POS) & LOW_LEFT_CHILD_MASK);
    }
    void SetLeftChild(bool exists) {
        _customData_nodeType = (_customData_nodeType & (~LEFT_CHILD_MASK)) | (((uint64_t)exists) << LEFT_CHILD_POS);
    }
    int64_t GetRightChildIndex() const {
        return (int64_t)(_customData_nodeType >> RIGHT_CHILD_POS);
    }
    void SetRightChildIndex(int64_t i) {
        _customData_nodeType = (_customData_nodeType & (~RIGHT_CHILD_MASK)) | (i << RIGHT_CHILD_POS);
    }
};

// data up to 10^8
// build from left
// size 8 bytes
class SplitNode : public InnerNode
{
private:
    // 58 bits right child index, 1 bit left child, 3 bits dimension, 2 bits node type
    //uint64_t rightChild_leftChild_splitDim_nodeType;
public:
    SplitNode(int splitDim) : InnerNode(NodeType::SPLIT) { SetSplitDim(splitDim); }

    int GetSplitDim() const {
        return (int)((_customData_nodeType >> DIM_POS) & LOW_DIM_MASK);
    }
private:
    void SetSplitDim(int splitDim) {
        _customData_nodeType = (_customData_nodeType & (~DIM_MASK)) | (((uint64_t)splitDim) << DIM_POS);
    }
};

// size 8 + 2*sizeof(Vec<FloatT, Dim>) bytes
// float:  8 +  8*Dim
// double: 8 + 16*Dim
template<typename FloatT, int Dim>
class ShrinkNode : public InnerNode
{
private:
    //uint64_t rightChild_leftChild_nodeType;
    Box<FloatT, Dim> _shrinkBox;
public:
    ShrinkNode(const Box<FloatT, Dim>& shrinkBox) : InnerNode(NodeType::SHRINK), _shrinkBox(shrinkBox) { }

    const Box<FloatT, Dim>& GetShrinkBox() const { return _shrinkBox; }
};

// size 16 bytes
class LeafNode : public Node
{
private:
    uint64_t _pointsEnd = 0;
public:
    LeafNode() : Node(NodeType::LEAF) { }

    int64_t GetPointsBegIndex() const { return (int64_t)(_customData_nodeType >> NODE_TYPE_BITS); }
    void SetPointsBegIndex(uint64_t i) { _customData_nodeType = (_customData_nodeType & NODE_TYPE_MASK) | (i << NODE_TYPE_BITS); }
    int64_t GetPointsEndIndex() const { return _pointsEnd; }
    void SetPointsEndIndex(uint64_t i) { _pointsEnd = i; }
};

template<typename FloatT, int Dim>
int GetNodeOffset(NodeType nodeType)
{
    switch (nodeType)
    {
    case NodeType::SPLIT:
        return sizeof(SplitNode) / sizeof(Node);
    case NodeType::SHRINK:
        return sizeof(ShrinkNode<FloatT, Dim>) / sizeof(Node);
    case NodeType::LEAF:
        return sizeof(LeafNode) / sizeof(Node);
    default:
        break;
    }
    return 0;
}

template<typename FloatT, int Dim, typename FuncT>
Vec<FloatT, Dim>* SplitPoints(Vec<FloatT, Dim>* beg, Vec<FloatT, Dim>* end, Vec<FloatT, Dim>* auxBeg, FuncT isLeft)
{
    int size = end - beg;
    int left = 0;
    int right = size - 1;
    for (const Vec<FloatT, Dim>* point = beg; point != end; ++point)
    {
        bool putLeft = isLeft(*point);
        if (putLeft) {
            auxBeg[left++] = *point;
        } else {
            auxBeg[right--] = *point;
        }
    }
    std::copy(auxBeg, auxBeg + size, beg);
    return beg + left;
}

template<typename FloatT, int Dim>
class BBDTree
{
public:
    BBDTree() {}

    static BBDTree BuildFairSplitTree(int leafMaxSize, const std::vector<Vec<FloatT, Dim>>& points)
    {
        BBDTree tree;
        tree._pointsBox = Box<FloatT, Dim>::GetBoundingBox(points);
        _leafMaxSize = leafMaxSize;
        _points = points;
        std::vector<Vec<FloatT, Dim>> aux = points;
        tree.BuildFairSplitTreeR(tree._pointsBox, _points.data(), _points.data() + _points.size(), aux.data());
        return tree;
    }
    static BBDTree BuildMidpointSplitTree(int leafMaxSize, const std::vector<Vec<FloatT, Dim>>& points)
    {
        BBDTree tree;
        tree._pointsBox = Box<FloatT, Dim>::GetBoundingBox(points);

        return tree;
    }

    Node* GetRoot() { return GetNode(0); }
    Node* GetNode(int64_t index) { return &nodes[index]; }
    
    const Node* GetRoot() const { return GetNode(0); }
    const Node* GetNode(int64_t index) const { return &nodes[index]; }

private:
    std::vector<Node> _nodes;
    std::vector<Vec<FloatT, Dim>> _points;
    Box<FloatT, Dim> _pointsBox;
    int _leafMaxSize;

    SplitNode* AddSplitNode(int splitDim);
    ShrinkNode* AddShrinkNode(const Box<FloatT, Dim>& shrinkBox);
    LeafNode* AddLeafNode(uint64_t pointsFrom, uint64_t pointsTo);

    Node* BuildFairSplitTreeR(const Box<FloatT, Dim>& box, Vec<FloatT, Dim>* pointsBeg, Vec<FloatT, Dim>* pointsEnd, Vec<FloatT, Dim>* auxBeg)
    {
        int size = pointsEnd - pointsBeg;
        if (size == 0) {
            return nullptr;
        }
        else if (size <= _leafMaxSize) {
            return AddLeafNode(pointsBeg - _points.data(), pointsEnd - _points.data());
        } else {
            BoxSplit<FloatT> split = box.FairSplit();
            SplitNode* splitNode = AddSplitNode(split.dim);
            Vec<FloatT, Dim>* splitTo = SplitPoints(pointsBeg, pointsEnd, auxBeg, [&split](const Vec<FloatT, Dim>& point) {
                return point[split.dim] < split.value;
            });
            // build left subtree
            Node* leftChild = BuildFairSplitTreeR(split.left, pointsBeg, splitTo, auxBeg);
            if (leftChild) {
                splitNode->SetLeftChild(true);
            }
            // build right subtree
            Node* rightChild = BuildFairSplitTreeR(split.right, splitTo, pointsEnd, auxBeg + (pointsBeg - splitTo));
            if (rightChild) {
                splitNode->SetRightChildIndex(rightChild - GetRoot());
            }
            return splitNode;
        }
    }
};

#endif // AKNN_BBD_TREE_H
