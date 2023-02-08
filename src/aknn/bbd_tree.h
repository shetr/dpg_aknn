#ifndef AKNN_BBD_TREE_H
#define AKNN_BBD_TREE_H

#include <vector>
#include <stdint.h>

#include "vec.h"

//! Enum used to distinguish between node types inside the BBD tree
enum class NodeType
{
    //! Split inner node
    SPLIT = 0,
    //! Shrink inner node
    SHRINK,
    //! Leaf node
    LEAF
};

//! Index type for normal-sized datasets, up to 10^8
using index_t = uint32_t;
//! Index type for datasets of size greather than 10^8
//using index_t = uint64_t;

// Definitions used for specifying the binary representation of nodes inside BBD tree

// Numbers of bits used for each section
#define NODE_TYPE_BITS 2
#define DIM_BITS 2
#define LEFT_CHILD_BITS 1
#define RIGHT_CHILD_BITS (sizeof(index_t) - (LEFT_CHILD_BITS + DIM_BITS + NODE_TYPE_BITS))

// Positions of sections inside the binary representation
#define DIM_POS (NODE_TYPE_BITS)
#define LEFT_CHILD_POS (DIM_BITS + NODE_TYPE_BITS)
#define RIGHT_CHILD_POS (LEFT_CHILD_BITS + DIM_BITS + NODE_TYPE_BITS)

// Masks of sections inside the binary representation
#define NODE_TYPE_MASK ((1 << NODE_TYPE_BITS) - 1)

#define LOW_DIM_MASK ((1 << DIM_BITS) - 1)
#define LOW_LEFT_CHILD_MASK ((1 << LEFT_CHILD_BITS) - 1)
#define LOW_RIGHT_CHILD_MASK ((1 << RIGHT_CHILD_BITS) - 1)

#define DIM_MASK (((1 << DIM_BITS) - 1) << DIM_POS)
#define LEFT_CHILD_MASK (((1 << LEFT_CHILD_BITS) - 1) << LEFT_CHILD_POS)
#define RIGHT_CHILD_MASK (((1 << RIGHT_CHILD_BITS) - 1) << RIGHT_CHILD_POS)

//! Base class for all nodes inside the BBD tree
class Node
{
protected:
    //! First lower 2 bits store NodeType value, remaining bits are used differently depending on the node type.
    index_t _customData_nodeType;
public:
    //! Default initialization for empty nodes
    Node() : _customData_nodeType(0) {}
    //! Initialize to specified node type, should be called by all derived nodes
    Node(NodeType nodeType) : _customData_nodeType(static_cast<index_t>(nodeType)) {}
    //! Gets the node type
    NodeType GetType() const { return static_cast<NodeType>(_customData_nodeType & NODE_TYPE_MASK); };
};

//! Base class for all inner nodes of the BBD tree
//! The _customData_nodeType variable now should always have the following format:
//! (sizeof(index_t) - 5) bits - right child index | 1 bit - has left child | 2 bits - dimension (only split node) | 2 bits - NodeType
class InnerNode : public Node
{
public:
    //! Initialize to specified node type, should be called by all derived nodes
    InnerNode(NodeType nodeType) : Node(nodeType) {}

    //! True if node has left child
    bool HasLeftChild() const {
        return (bool)((_customData_nodeType >> LEFT_CHILD_POS) & LOW_LEFT_CHILD_MASK);
    }
    //! Set if node has left child or not
    void SetLeftChild(bool exists) {
        _customData_nodeType = (_customData_nodeType & (~LEFT_CHILD_MASK)) | (((index_t)exists) << LEFT_CHILD_POS);
    }
    //! Get index of the right child. Index 0 means that the node doesn't have right child.
    index_t GetRightChildIndex() const {
        return (index_t)(_customData_nodeType >> RIGHT_CHILD_POS);
    }
    //! Set index of the right child. Index 0 means that the node doesn't have right child.
    void SetRightChildIndex(index_t i) {
        _customData_nodeType = (_customData_nodeType & (~RIGHT_CHILD_MASK)) | (i << RIGHT_CHILD_POS);
    }
};

//! Node representing split of current bounding box in half in some specified dimension.
//! The node uses the 2 dimension bits from _customData_nodeType.
//! Size of the node is 4 or 8 bytes depending on the index_t (default 4 bytes).
class SplitNode : public InnerNode
{
public:
    //! Create split node in specified dimension
    SplitNode(int splitDim) : InnerNode(NodeType::SPLIT) { SetSplitDim(splitDim); }
    //! Gets split dimension
    int GetSplitDim() const {
        return (int)((_customData_nodeType >> DIM_POS) & LOW_DIM_MASK);
    }
private:
    void SetSplitDim(int splitDim) {
        _customData_nodeType = (_customData_nodeType & (~DIM_MASK)) | (((index_t)splitDim) << DIM_POS);
    }
};

//! Node dividing current box into inner and outer part with some shrink box representing the inner part.
//! Outer part is everything else insed the current box.
//! Left child is the inner part and right child is the outer part.
//! Size of the node is sizof(index_t) + 2 * Dim * sizeof(FloatT) B. So for 4-byte index_t and FloatT the size should be at least:
//! Dim = 2: size = 20 B; Dim = 3: size = 28 B ; Dim = 4: size = 36 B
template<typename FloatT, int Dim>
class ShrinkNode : public InnerNode
{
private:
    //! Box representing the inner part
    Box<FloatT, Dim> _shrinkBox;
public:
    //! Initialize with specified shrink box
    ShrinkNode(const Box<FloatT, Dim>& shrinkBox) : InnerNode(NodeType::SHRINK), _shrinkBox(shrinkBox) { }
    //! Gets the shrink box
    const Box<FloatT, Dim>& GetShrinkBox() const { return _shrinkBox; }
};

//! Leaf node, referencing some range of objects. It stores begin and end indices to the array of objects.
//! Begin index is stored inside upper (sizeof(index_t) - 2) bits of _customData_nodeType. End index is stored in new variable _objsEnd.
//! Size of the node is 8 or 16 bytes depending on the index_t (default 8 bytes).
class LeafNode : public Node
{
private:
    //! End index of the referenced objects range
    index_t _objsEnd = 0;
public:
    //! Initialize with range of point objects indices
    LeafNode(index_t pointsBeg, index_t pointsEnd) : Node(NodeType::LEAF) {
        SetPointsBegIndex(pointsBeg);
        SetPointsEndIndex(pointsEnd);
    }

    //! Get begin index
    index_t GetPointsBegIndex() const { return (index_t)(_customData_nodeType >> NODE_TYPE_BITS); }
    //! Get end index
    index_t GetPointsEndIndex() const { return _objsEnd; }
private:
    void SetPointsBegIndex(index_t i) { _customData_nodeType = (_customData_nodeType & NODE_TYPE_MASK) | (i << NODE_TYPE_BITS); }
    void SetPointsEndIndex(index_t i) { _objsEnd = i; }
};

//! Gets offset of specified node type inside array of Nodes (multiples of sizeof(Node))
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

//! Splits array into 2 parts (like in quick sort) according to FuncT isLeft function.
//! FuncT has 2 parameters and should return true if left parameter is "smaller" than right parameter.
//! Returns pointer to begining of the right part (end of left part).
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

//! Represents current state when building the BBD tree.
template<typename FloatT, int Dim, typename ObjData>
struct SplitState
{
    Box<FloatT, Dim> box;
    PointObj<FloatT, Dim, ObjData>* pointsBeg;
    PointObj<FloatT, Dim, ObjData>* pointsEnd;

    index_t size() const { return pointsEnd - pointsBeg; };
};

//! Various statistics of the BBD tree
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

// Intermediate structure for computing statistics of the BBD tree
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

//! BBDTree consits of 2 types of inner nodes: split nodes and shrink nodes. Each is described within their own type.
//! This class contains functions for building the tree, accessing its nodes and getting some basic statistics.
template<typename FloatT, int Dim, typename ObjData = Empty>
class BBDTree
{
public:
    //! Definition to simplify the code
    using PointObjT = PointObj<FloatT, Dim, ObjData>;

    //! Initializes empty tree
    BBDTree() {}
    //! Builds the tree using only splits (used only for testing purposes)
    static BBDTree BuildBasicSplitTree(int leafMaxSize, const std::vector<PointObjT>& objs)
    {
        BBDTree tree(leafMaxSize, objs);
        tree.BuildBasicSplitTreeR({tree._bbox, tree._objs.data(), tree._objs.data() + tree._objs.size()});
        return tree;
    }
    //! Builds the tree using midpoint split algorithm. Result has both split and shrink nodes.
    static BBDTree BuildMidpointSplitTree(int leafMaxSize, const std::vector<PointObjT>& objs)
    {
        BBDTree tree(leafMaxSize, objs);
        tree.BuildMidpointSplitTreeR({tree._bbox, tree._objs.data(), tree._objs.data() + tree._objs.size()});
        return tree;
    }

    //! Gets the root node
    Node* GetRoot() { return GetNode(0); }
    //! Gets node by index
    Node* GetNode(index_t index) { return &_nodes[index]; }
    
    //! Gets the root node, read only
    const Node* GetRoot() const { return GetNode(0); }
    //! Gets node by index, read only
    const Node* GetNode(index_t index) const { return &_nodes[index]; }
    
    //! Gets point object by index, read only
    const PointObjT* GetObj(index_t index) const { return _objs.data() + index; }

    //! Gets bounding box of all the points.
    const Box<FloatT, Dim>& GetBBox() const { return _bbox; }

    //! Gets number of point objects referenced by leaf node
    int GetLeafSize(const LeafNode* leafNode) const {
        return GetObj(leafNode->GetPointsEndIndex()) - GetObj(leafNode->GetPointsBegIndex());
    }

    //! Gets tree statistics
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
    //! Array of inner and leaf nodes. The actual nodes are written in an "unsafe" way.
    //! Depending on the node type the node may span on multiple Node elements. For example when sizeof(index_t)=4 and sizeof(FloatT)=4:
    //! Then SplitNode is 1 * Node, LeafNode is 2 * Node, ShrinkNode is 7 * Node for Dim = 3
    std::vector<Node> _nodes;
    //! Source point objects for which the search is optimized
    std::vector<PointObjT> _objs;
    //! Bounding box of the point objects
    Box<FloatT, Dim> _bbox;
    //! Max leaf size
    int _leafMaxSize;

    //! Initialization before building the tree
    BBDTree(int leafMaxSize, const std::vector<PointObjT>& objs) : _leafMaxSize(leafMaxSize), _objs(objs), _bbox(Box<FloatT, Dim>::GetBoundingBox(objs)) {}

    //! Adds SplitNode to nodes array and returns its index
    index_t AddSplitNode(int splitDim) {
        static_assert(sizeof(SplitNode) == sizeof(Node));
        index_t index = (int64_t)_nodes.size();
        _nodes.push_back(Node());
        SplitNode* splitNode = (SplitNode*)&_nodes.back();
        *splitNode = SplitNode(splitDim);
        return index;
    }
    //! Adds ShrinkNode to nodes array and returns its index
    index_t AddShrinkNode(const Box<FloatT, Dim>& shrinkBox) {
        static_assert(sizeof(ShrinkNode<FloatT, Dim>) % sizeof(Node) == 0);
        index_t index = (int64_t)_nodes.size();
        _nodes.resize(_nodes.size() + GetNodeOffset<FloatT, Dim>(NodeType::SHRINK));
        ShrinkNode<FloatT, Dim>* shrinkNode = (ShrinkNode<FloatT, Dim>*) &_nodes[index];
        *shrinkNode = ShrinkNode<FloatT, Dim>(shrinkBox);
        return index;
    }
    //! Adds LeafNode to nodes array and returns its index
    index_t AddLeafNode(index_t pointsBeg, index_t pointsEnd) {
        static_assert(sizeof(LeafNode) == 2*sizeof(Node));
        index_t index = (int64_t)_nodes.size();
        _nodes.resize(_nodes.size() + GetNodeOffset<FloatT, Dim>(NodeType::LEAF));
        LeafNode* leafNode = (LeafNode*)&_nodes[index];
        *leafNode = LeafNode(pointsBeg, pointsEnd);
        return index;
    }

    //! Builds child subtrees using only splits (used only for testing purposes)
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

    //! Builds the tree using only splits (used only for testing purposes)
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

    //! Builds the child subtrees of parent node using midpoint split algorithm.
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

    //! Splits current bounding box and remembers the part containing more points
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

    //! Builds the tree using midpoint split algorithm recursively.
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
            // split until number of points is <= 2/3 total number of points in current node
            while (3 * splitState.size() > 2 * state.size() && splitState.size() > _leafMaxSize) {
                SetBiggerSplit(splitState, split, splitTo);
                ++splitCount;
            }
            // if there was only 1 split, we can simply create split node
            if (splitCount == 1) {
                index_t splitNodeIndex = AddSplitNode(split.dim);
                BuildMidpointSplitChilds(splitNodeIndex, split.left, split.right, state.pointsBeg, state.pointsEnd, splitTo);
                return splitNodeIndex;
            } else { // otherwise we have a shrink node
                // split the points according to the final box (inside box is on left, outside on right)
                PointObjT* insideBoxTo = SplitPoints(state.pointsBeg, state.pointsEnd, [&splitState](const Vec<FloatT, Dim>& point) {
                    return splitState.box.Includes(point);
                });
                index_t shrinkNodeIndex = AddShrinkNode(splitState.box);
                BuildMidpointSplitChilds(shrinkNodeIndex, splitState.box, state.box, state.pointsBeg, state.pointsEnd, insideBoxTo);
                return shrinkNodeIndex;
            }
        }
    }

    //! Gets tree statistics recursively
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

#endif // AKNN_BBD_TREE_H
