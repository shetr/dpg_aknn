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

// index type for normal-sized datasets, up to 10^8
using index_t = uint32_t;
// index type for datasets of size greather than 10^8
//using index_t = uint64_t;

#define NODE_TYPE_BITS 2
#define DIM_BITS 2
#define LEFT_CHILD_BITS 1
#define RIGHT_CHILD_BITS (sizeof(index_t) - (LEFT_CHILD_BITS + DIM_BITS + NODE_TYPE_BITS))

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

/*index_t GetBits(index_t storage, index_t bitPos, index_t lowMask) {
    return (storage >> bitPos) & lowMask;
}

void SetBits(index_t& storage, index_t bits, index_t bitPos, index_t mask) {
    storage = (storage & (~mask)) | (bits << bitPos);
}*/

class Node
{
protected:
    index_t _customData_nodeType;
public:
    Node() : _customData_nodeType(0) {}
    Node(NodeType nodeType) : _customData_nodeType(static_cast<index_t>(nodeType)) {}

    NodeType GetType() const { return static_cast<NodeType>(_customData_nodeType & NODE_TYPE_MASK); };
};

class InnerNode : public Node
{
public:
    InnerNode(NodeType nodeType) : Node(nodeType) {}

    bool HasLeftChild() const {
        return (bool)((_customData_nodeType >> LEFT_CHILD_POS) & LOW_LEFT_CHILD_MASK);
    }
    void SetLeftChild(bool exists) {
        _customData_nodeType = (_customData_nodeType & (~LEFT_CHILD_MASK)) | (((index_t)exists) << LEFT_CHILD_POS);
    }
    index_t GetRightChildIndex() const {
        return (index_t)(_customData_nodeType >> RIGHT_CHILD_POS);
    }
    void SetRightChildIndex(index_t i) {
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
    //index_t rightChild_leftChild_splitDim_nodeType;
public:
    SplitNode(int splitDim) : InnerNode(NodeType::SPLIT) { SetSplitDim(splitDim); }

    int GetSplitDim() const {
        return (int)((_customData_nodeType >> DIM_POS) & LOW_DIM_MASK);
    }
private:
    void SetSplitDim(int splitDim) {
        _customData_nodeType = (_customData_nodeType & (~DIM_MASK)) | (((index_t)splitDim) << DIM_POS);
    }
};

// size 8 + 2*sizeof(Vec<FloatT, Dim>) bytes
// float:  8 +  8*Dim
// double: 8 + 16*Dim
template<typename FloatT, int Dim>
class ShrinkNode : public InnerNode
{
private:
    //index_t rightChild_leftChild_nodeType;
    Box<FloatT, Dim> _shrinkBox;
public:
    ShrinkNode(const Box<FloatT, Dim>& shrinkBox) : InnerNode(NodeType::SHRINK), _shrinkBox(shrinkBox) { }

    const Box<FloatT, Dim>& GetShrinkBox() const { return _shrinkBox; }
};

// size 16 bytes
class LeafNode : public Node
{
private:
    index_t _objsEnd = 0;
public:
    LeafNode(index_t pointsBeg, index_t pointsEnd) : Node(NodeType::LEAF) {
        SetPointsBegIndex(pointsBeg);
        SetPointsEndIndex(pointsEnd);
    }

    index_t GetPointsBegIndex() const { return (index_t)(_customData_nodeType >> NODE_TYPE_BITS); }
    index_t GetPointsEndIndex() const { return _objsEnd; }
private:
    void SetPointsBegIndex(index_t i) { _customData_nodeType = (_customData_nodeType & NODE_TYPE_MASK) | (i << NODE_TYPE_BITS); }
    void SetPointsEndIndex(index_t i) { _objsEnd = i; }
};

template<typename FloatT, int Dim>
index_t GetNodeOffset(NodeType nodeType)
{
    static_assert(sizeof(SplitNode) == sizeof(Node));
    static_assert(sizeof(InnerNode) == sizeof(Node));
    static_assert(sizeof(ShrinkNode<FloatT, Dim>) % sizeof(Node) == 0);
    static_assert(sizeof(LeafNode) == 2*sizeof(Node));
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

template<typename FloatT, int Dim, typename FuncT, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData>* SplitPoints(PointObj<FloatT, Dim, ObjData>* beg, PointObj<FloatT, Dim, ObjData>* end, FuncT isLeft)
{
    int size = end - beg;
    int left = 0;
    int right = size - 1;
    while (left <= right) {
        bool putLeft = isLeft(beg[left].point);
        if (putLeft) {
            ++left;
        } else {
            std::swap(beg[left], beg[right--]);
        }
    }
    return beg + left;
}

template<typename FloatT, int Dim, typename ObjData>
struct SplitState
{
    Box<FloatT, Dim> box;
    PointObj<FloatT, Dim, ObjData>* pointsBeg;
    PointObj<FloatT, Dim, ObjData>* pointsEnd;

    index_t size() const { return pointsEnd - pointsBeg; };
};

struct BBDTreeStats
{
    int innerNodeCount;
    int leafNodeCount;
    int splitNodeCount;
    int shrinkNodeCount;
    int nullCount;
    int maxDepth;
    double avgDepth;
    double avgLeafSize;
    // in bytes
    int memoryConsumption;
};

struct BBDTreeIntermediateStats
{
    int innerNodeCount = 0;
    int leafNodeCount = 0;
    int splitNodeCount = 0;
    int shrinkNodeCount = 0;
    int nullCount = 0;
    int maxDepth = 0;
    int depthSum = 0;
    int leafSizesSum = 0;
};

template<typename FloatT, int Dim, typename ObjData = Empty>
class BBDTree
{
public:
    using PointObjT = PointObj<FloatT, Dim, ObjData>;

    BBDTree() {}

    static BBDTree BuildBasicSplitTree(int leafMaxSize, const std::vector<PointObjT>& objs)
    {
        BBDTree tree(leafMaxSize, objs);
        tree.BuildBasicSplitTreeR({tree._bbox, tree._objs.data(), tree._objs.data() + tree._objs.size()});
        return tree;
    }
    static BBDTree BuildMidpointSplitTree(int leafMaxSize, const std::vector<PointObjT>& objs)
    {
        BBDTree tree(leafMaxSize, objs);
        tree.BuildMidpointSplitTreeR({tree._bbox, tree._objs.data(), tree._objs.data() + tree._objs.size()});
        return tree;
    }

    // TODO: handle edge case when there are 0 nodes (return nullptr)
    Node* GetRoot() { return GetNode(0); }
    Node* GetNode(index_t index) { return &_nodes[index]; }
    
    const Node* GetRoot() const { return GetNode(0); }
    const Node* GetNode(index_t index) const { return &_nodes[index]; }
    
    const PointObjT* GetObj(index_t index) const { return _objs.data() + index; }

    const Box<FloatT, Dim>& GetBBox() const { return _bbox; }

    int GetLeafSize(const LeafNode* leafNode) const {
        return GetObj(leafNode->GetPointsEndIndex()) - GetObj(leafNode->GetPointsBegIndex());
    }

    BBDTreeStats GetStats() const {
        BBDTreeIntermediateStats interStats;
        GetStatsR(interStats, 0, 0);
        BBDTreeStats stats;
        stats.innerNodeCount = interStats.innerNodeCount;
        stats.leafNodeCount = interStats.leafNodeCount;
        stats.splitNodeCount = interStats.splitNodeCount;
        stats.shrinkNodeCount = interStats.shrinkNodeCount;
        stats.nullCount = interStats.nullCount;
        stats.maxDepth = interStats.maxDepth;
        stats.avgDepth = (double)interStats.depthSum / (double)interStats.leafNodeCount;
        stats.avgLeafSize = (double)interStats.leafSizesSum / (double)interStats.leafNodeCount;
        stats.memoryConsumption = sizeof(Node) * _nodes.size();
        return stats;
    }
private:
    std::vector<Node> _nodes;
    std::vector<PointObjT> _objs;
    Box<FloatT, Dim> _bbox;
    int _leafMaxSize;

    BBDTree(int leafMaxSize, const std::vector<PointObjT>& objs) : _leafMaxSize(leafMaxSize), _objs(objs), _bbox(Box<FloatT, Dim>::GetBoundingBox(objs)) {}

    index_t AddSplitNode(int splitDim) {
        static_assert(sizeof(SplitNode) == sizeof(Node));
        index_t index = (int64_t)_nodes.size();
        _nodes.push_back(Node());
        SplitNode* splitNode = (SplitNode*)&_nodes.back();
        *splitNode = SplitNode(splitDim);
        return index;
    }
    index_t AddShrinkNode(const Box<FloatT, Dim>& shrinkBox) {
        static_assert(sizeof(ShrinkNode<FloatT, Dim>) % sizeof(Node) == 0);
        index_t index = (int64_t)_nodes.size();
        _nodes.resize(_nodes.size() + GetNodeOffset<FloatT, Dim>(NodeType::SHRINK));
        ShrinkNode<FloatT, Dim>* shrinkNode = (ShrinkNode<FloatT, Dim>*) &_nodes[index];
        *shrinkNode = ShrinkNode<FloatT, Dim>(shrinkBox);
        return index;
    }
    index_t AddLeafNode(index_t pointsBeg, index_t pointsEnd) {
        static_assert(sizeof(LeafNode) == 2*sizeof(Node));
        index_t index = (int64_t)_nodes.size();
        _nodes.resize(_nodes.size() + GetNodeOffset<FloatT, Dim>(NodeType::LEAF));
        LeafNode* leafNode = (LeafNode*)&_nodes[index];
        *leafNode = LeafNode(pointsBeg, pointsEnd);
        return index;
    }

    void BuildBasicSplitChilds(index_t parentIndex, const Box<FloatT, Dim>& leftBox, const Box<FloatT, Dim>& rightBox, PointObjT* pointsBeg, PointObjT* pointsEnd, PointObjT* splitTo)
    {
        // build left subtree
        index_t leftChildIndex = BuildBasicSplitTreeR({leftBox, pointsBeg, splitTo});
        if (leftChildIndex) {
            InnerNode* parentNode = (InnerNode*) &_nodes[parentIndex];
            parentNode->SetLeftChild(true);
        }
        // build right subtree
        index_t rightChildIndex = BuildBasicSplitTreeR({rightBox, splitTo, pointsEnd});
        if (rightChildIndex) {
            InnerNode* parentNode = (InnerNode*) &_nodes[parentIndex];
            parentNode->SetRightChildIndex(rightChildIndex);
        }
    }

    index_t BuildBasicSplitTreeR(SplitState<FloatT, Dim, ObjData> state)
    {
        if (state.size() == 0) {
            // indicate that there is no child with 0 index (only root has 0 index and it can't be child of any node)
            return 0;
        } else if (state.size() <= _leafMaxSize) {
            // add leaf node if number of points is small enough
            return AddLeafNode(state.pointsBeg - _objs.data(), state.pointsEnd - _objs.data());
        } else {
            BoxSplit<FloatT, Dim> split = state.box.Split();
            index_t splitNodeIndex = AddSplitNode(split.dim);
            // split points
            PointObjT* splitTo = SplitPoints(state.pointsBeg, state.pointsEnd, [&split](const Vec<FloatT, Dim>& point) {
                return point[split.dim] < split.value;
            });
            BuildBasicSplitChilds(splitNodeIndex, split.left, split.right, state.pointsBeg, state.pointsEnd, splitTo);
            return splitNodeIndex;
        }
    }

    void BuildMidpointSplitChilds(index_t parentIndex, const Box<FloatT, Dim>& leftBox, const Box<FloatT, Dim>& rightBox, PointObjT* pointsBeg, PointObjT* pointsEnd, PointObjT* splitTo)
    {
        // build left subtree
        index_t leftChildIndex = BuildMidpointSplitTreeR({leftBox, pointsBeg, splitTo});
        if (leftChildIndex) {
            InnerNode* parentNode = (InnerNode*) &_nodes[parentIndex];
            parentNode->SetLeftChild(true);
        }
        // build right subtree
        index_t rightChildIndex = BuildMidpointSplitTreeR({rightBox, splitTo, pointsEnd});
        if (rightChildIndex) {
            InnerNode* parentNode = (InnerNode*) &_nodes[parentIndex];
            parentNode->SetRightChildIndex(rightChildIndex);
        }
    }

    void SetBiggerSplit(SplitState<FloatT, Dim, ObjData>& state, BoxSplit<FloatT, Dim>& split, PointObjT*& splitTo)
    {
        split = state.box.Split();
        splitTo = SplitPoints(state.pointsBeg, state.pointsEnd, [&split](const Vec<FloatT, Dim>& point) {
            return point[split.dim] < split.value;
        });
        index_t leftSize = splitTo - state.pointsBeg;
        index_t rightSize = state.pointsEnd - splitTo;
        bool leftBigger = leftSize >= rightSize;
        if (leftBigger) {
            state.box = split.left;
            state.pointsEnd = splitTo;
        } else {
            state.box = split.right;
            state.pointsBeg = splitTo;
        }
    }

    index_t BuildMidpointSplitTreeR(SplitState<FloatT, Dim, ObjData> state)
    {
        if (state.size() == 0) {
            // indicate that there is no child with 0 index (only root has 0 index and it can't be child of any node)
            return 0;
        } else if (state.size() <= _leafMaxSize) {
            // add leaf node if number of points is small enough
            return AddLeafNode(state.pointsBeg - _objs.data(), state.pointsEnd - _objs.data());
        } else {
            BoxSplit<FloatT, Dim> split;
            PointObjT* splitTo;
            SplitState<FloatT, Dim, ObjData> splitState = state;
            int splitCount = 0;
            while (3 * splitState.size() > 2 * state.size() && splitState.size() > _leafMaxSize) {
                SetBiggerSplit(splitState, split, splitTo);
                ++splitCount;
            }
            if (splitCount == 1) {
                index_t splitNodeIndex = AddSplitNode(split.dim);
                BuildMidpointSplitChilds(splitNodeIndex, split.left, split.right, state.pointsBeg, state.pointsEnd, splitTo);
                return splitNodeIndex;
            } else {
                PointObjT* insideBoxTo = SplitPoints(state.pointsBeg, state.pointsEnd, [&splitState](const Vec<FloatT, Dim>& point) {
                    return splitState.box.Includes(point);
                });
                index_t shrinkNodeIndex = AddShrinkNode(splitState.box);
                BuildMidpointSplitChilds(shrinkNodeIndex, splitState.box, state.box, state.pointsBeg, state.pointsEnd, insideBoxTo);
                return shrinkNodeIndex;
            }
        }
    }

    void GetStatsR(BBDTreeIntermediateStats& stats, index_t nodeIndex, int depth) const {
        const Node* node = GetNode(nodeIndex);

        if (node->GetType() == NodeType::LEAF)
        {
            const LeafNode* leafNode = (const LeafNode*)node;
            stats.leafSizesSum += GetLeafSize(leafNode);
            ++stats.leafNodeCount;
            stats.depthSum += depth;
            stats.maxDepth = std::max(stats.maxDepth, depth);
        }
        else
        {
            const InnerNode* innerNode = (const InnerNode*)node;
            ++stats.innerNodeCount;
            if (node->GetType() == NodeType::SPLIT) {
                ++stats.splitNodeCount;
            }
            else if (node->GetType() == NodeType::SHRINK) {
                ++stats.shrinkNodeCount;
            }

            if (innerNode->HasLeftChild())
                GetStatsR(stats, nodeIndex + GetNodeOffset<FloatT, Dim>(node->GetType()), depth + 1);
            else
                ++stats.nullCount;
            
            if (innerNode->GetRightChildIndex() != 0)
                GetStatsR(stats, innerNode->GetRightChildIndex(), depth + 1);
            else
                ++stats.nullCount;
        }
    }
};

//---------------------------------------------------------------------------------------------------------------------------------
// Implemetation
//---------------------------------------------------------------------------------------------------------------------------------


#endif // AKNN_BBD_TREE_H
