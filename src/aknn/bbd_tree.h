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
    Node() : _customData_nodeType(0) {}
    Node(NodeType nodeType) : _customData_nodeType(static_cast<uint64_t>(nodeType)) {}

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
        _customData_nodeType = (_customData_nodeType & (~LEFT_CHILD_MASK)) | (((uint64_t)exists) << LEFT_CHILD_POS);
    }
    uint64_t GetRightChildIndex() const {
        return (uint64_t)(_customData_nodeType >> RIGHT_CHILD_POS);
    }
    void SetRightChildIndex(uint64_t i) {
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
    uint64_t _objsEnd = 0;
public:
    LeafNode(uint64_t pointsBeg, uint64_t pointsEnd) : Node(NodeType::LEAF) {
        SetPointsBegIndex(pointsBeg);
        SetPointsEndIndex(pointsEnd);
    }

    uint64_t GetPointsBegIndex() const { return (uint64_t)(_customData_nodeType >> NODE_TYPE_BITS); }
    uint64_t GetPointsEndIndex() const { return _objsEnd; }
private:
    void SetPointsBegIndex(uint64_t i) { _customData_nodeType = (_customData_nodeType & NODE_TYPE_MASK) | (i << NODE_TYPE_BITS); }
    void SetPointsEndIndex(uint64_t i) { _objsEnd = i; }
};

template<typename FloatT, int Dim>
uint64_t GetNodeOffset(NodeType nodeType)
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
PointObj<FloatT, Dim, ObjData>* SplitPoints(PointObj<FloatT, Dim, ObjData>* beg, PointObj<FloatT, Dim, ObjData>* end, PointObj<FloatT, Dim, ObjData>* auxBeg, FuncT isLeft)
{
    int size = end - beg;
    int left = 0;
    int right = size - 1;
    for (const PointObj<FloatT, Dim, ObjData>* obj = beg; obj != end; ++obj)
    {
        bool putLeft = isLeft(obj->point);
        if (putLeft) {
            auxBeg[left++] = *obj;
        } else {
            auxBeg[right--] = *obj;
        }
    }
    std::copy(auxBeg, auxBeg + size, beg);
    return beg + left;
}

template<typename FloatT, int Dim, typename ObjData = Empty>
class BBDTree
{
public:
    using PointObjT = PointObj<FloatT, Dim, ObjData>;

    BBDTree() {}

    static BBDTree BuildFairSplitTree(int leafMaxSize, const std::vector<PointObjT>& objs)
    {
        BBDTree tree(leafMaxSize, objs);
        std::vector<PointObjT> aux = objs;
        tree.BuildFairSplitTreeR(tree._bbox, tree._objs.data(), tree._objs.data() + tree._objs.size(), aux.data());
        tree.RemoveTrivialNodes();
        return tree;
    }
    static BBDTree BuildMidpointSplitTree(int leafMaxSize, const std::vector<PointObjT>& objs)
    {
        BBDTree tree(leafMaxSize, objs);
        std::vector<PointObjT> aux = objs;
        tree.BuildMidpointSplitTreeR(tree._bbox, tree._objs.data(), tree._objs.data() + tree._objs.size(), aux.data());
        tree.RemoveTrivialNodes();
        return tree;
    }

    // TODO: handle edge case when there are 0 nodes (return nullptr)
    Node* GetRoot() { return GetNode(0); }
    Node* GetNode(uint64_t index) { return &_nodes[index]; }
    
    const Node* GetRoot() const { return GetNode(0); }
    const Node* GetNode(uint64_t index) const { return &_nodes[index]; }
    
    const PointObjT* GetObj(uint64_t index) const { return _objs.data() + index; }

    const Box<FloatT, Dim>& GetBBox() const { return _bbox; }

private:
    std::vector<Node> _nodes;
    std::vector<PointObjT> _objs;
    Box<FloatT, Dim> _bbox;
    int _leafMaxSize;

    BBDTree(int leafMaxSize, const std::vector<PointObjT>& objs) : _leafMaxSize(leafMaxSize), _objs(objs), _bbox(Box<FloatT, Dim>::GetBoundingBox(objs)) {}

    uint64_t AddSplitNode(int splitDim) {
        static_assert(sizeof(SplitNode) == sizeof(Node));
        uint64_t index = (int64_t)_nodes.size();
        _nodes.push_back(Node());
        SplitNode* splitNode = (SplitNode*)&_nodes.back();
        *splitNode = SplitNode(splitDim);
        return index;
    }
    uint64_t AddShrinkNode(const Box<FloatT, Dim>& shrinkBox) {
        static_assert(sizeof(ShrinkNode<FloatT, Dim>) % sizeof(Node) == 0);
        uint64_t index = (int64_t)_nodes.size();
        for(int i = 0; i < GetNodeOffset<FloatT, Dim>(NodeType::SHRINK); ++i) {
            _nodes.push_back(Node());
        }
        ShrinkNode<FloatT, Dim>* shrinkNode = (ShrinkNode<FloatT, Dim>*) &_nodes[index];
        *shrinkNode = ShrinkNode<FloatT, Dim>(shrinkBox);
        return index;
    }
    uint64_t AddLeafNode(uint64_t pointsBeg, uint64_t pointsEnd) {
        static_assert(sizeof(LeafNode) == 2*sizeof(Node));
        uint64_t index = (int64_t)_nodes.size();
        _nodes.push_back(Node());
        _nodes.push_back(Node());
        LeafNode* leafNode = (LeafNode*)&_nodes[index];
        *leafNode = LeafNode(pointsBeg, pointsEnd);
        return index;
    }

    uint64_t BuildFairSplitTreeR(const Box<FloatT, Dim>& box, PointObjT* pointsBeg, PointObjT* pointsEnd, PointObjT* auxBeg)
    {
        uint64_t size = pointsEnd - pointsBeg;
        if (size == 0) {
            // indicate that there is no child with 0 index (only root has 0 index and it can't be child of any node)
            return 0;
        } else if (size <= _leafMaxSize) {
            return AddLeafNode(pointsBeg - _objs.data(), pointsEnd - _objs.data());
        } else {
            BoxSplit<FloatT, Dim> split = box.FairSplit();
            uint64_t splitNodeIndex = AddSplitNode(split.dim);
            // split points
            PointObjT* splitTo = SplitPoints(pointsBeg, pointsEnd, auxBeg, [&split](const Vec<FloatT, Dim>& point) {
                return point[split.dim] < split.value;
            });
            // build left subtree
            uint64_t leftChildIndex = BuildFairSplitTreeR(split.left, pointsBeg, splitTo, auxBeg);
            if (leftChildIndex) {
                SplitNode* splitNode = (SplitNode*) &_nodes[splitNodeIndex];
                splitNode->SetLeftChild(true);
            }
            // build right subtree
            uint64_t rightChildIndex = BuildFairSplitTreeR(split.right, splitTo, pointsEnd, auxBeg + (splitTo - pointsBeg));
            if (rightChildIndex) {
                SplitNode* splitNode = (SplitNode*) &_nodes[splitNodeIndex];
                splitNode->SetRightChildIndex(rightChildIndex);
            }
            return splitNodeIndex;
        }
    }

    uint64_t BuildMidpointSplitTreeR(const Box<FloatT, Dim>& box, PointObjT* pointsBeg, PointObjT* pointsEnd, PointObjT* auxBeg)
    {
        uint64_t size = pointsEnd - pointsBeg;
        if (size == 0) {
            // indicate that there is no child with 0 index (only root has 0 index and it can't be child of any node)
            return 0;
        } else if (size <= _leafMaxSize) {
            return AddLeafNode(pointsBeg - _objs.data(), pointsEnd - _objs.data());
        } else {
            Box<FloatT, Dim> currentBox = box;
            PointObjT* currentBeg = pointsBeg;
            PointObjT* currentEnd = pointsEnd;
            PointObjT* currentAuxBeg = auxBeg;
            uint64_t currentSize = size;
            for (int splitCount = 0; splitCount < Dim / 2; ++splitCount) {
                BoxSplit<FloatT, Dim> split = currentBox.FairSplit();
                PointObjT* splitTo = SplitPoints(currentBeg, currentEnd, currentAuxBeg, [&split](const Vec<FloatT, Dim>& point) {
                    return point[split.dim] < split.value;
                });
                uint64_t leftSize = splitTo - currentBeg;
                uint64_t rightSize = currentEnd - splitTo;
                bool left = leftSize >= rightSize;
                if (left) {
                    currentBox = split.left;
                    currentSize = leftSize;
                    currentEnd = splitTo;
                } else {
                    currentBox = split.right;
                    currentSize = rightSize;
                    currentBeg = splitTo;
                    currentAuxBeg = currentAuxBeg + leftSize;
                }
                if (currentSize <= _leafMaxSize) {
                    break;
                }
            }
            if (currentSize * 2 <= size) {
                BoxSplit<FloatT, Dim> split = box.FairSplit();
                uint64_t splitNodeIndex = AddSplitNode(split.dim);
                // split points
                PointObjT* splitTo = SplitPoints(pointsBeg, pointsEnd, auxBeg, [&split](const Vec<FloatT, Dim>& point) {
                    return point[split.dim] < split.value;
                });
                // build left subtree
                uint64_t leftChildIndex = BuildMidpointSplitTreeR(split.left, pointsBeg, splitTo, auxBeg);
                if (leftChildIndex) {
                    SplitNode* splitNode = (SplitNode*) &_nodes[splitNodeIndex];
                    splitNode->SetLeftChild(true);
                }
                // build right subtree
                uint64_t rightChildIndex = BuildMidpointSplitTreeR(split.right, splitTo, pointsEnd, auxBeg + (splitTo - pointsBeg));
                if (rightChildIndex) {
                    SplitNode* splitNode = (SplitNode*) &_nodes[splitNodeIndex];
                    splitNode->SetRightChildIndex(rightChildIndex);
                }
                return splitNodeIndex;
            } else {
                while (currentSize > (2 * size) / 3) {
                    BoxSplit<FloatT, Dim> split = currentBox.FairSplit();
                    PointObjT* splitTo = SplitPoints(currentBeg, currentEnd, currentAuxBeg, [&split](const Vec<FloatT, Dim>& point) {
                        return point[split.dim] < split.value;
                    });
                    uint64_t leftSize = splitTo - currentBeg;
                    uint64_t rightSize = currentEnd - splitTo;
                    bool left = leftSize >= rightSize;
                    if (left) {
                        currentBox = split.left;
                        currentSize = leftSize;
                        currentEnd = splitTo;
                    } else {
                        currentBox = split.right;
                        currentSize = rightSize;
                        currentBeg = splitTo;
                        currentAuxBeg = currentAuxBeg + leftSize;
                    }
                    if (currentSize <= _leafMaxSize) {
                        break;
                    }
                }
                PointObjT* insideBoxTo = SplitPoints(pointsBeg, pointsEnd, auxBeg, [&currentBox](const Vec<FloatT, Dim>& point) {
                    return currentBox.Includes(point);
                });
                uint64_t shrinkNodeIndex = AddShrinkNode(currentBox);
                // build left subtree, inner box
                uint64_t leftChildIndex = BuildMidpointSplitTreeR(currentBox, pointsBeg, insideBoxTo, auxBeg);
                if (leftChildIndex) {
                    InnerNode* node = (InnerNode*) &_nodes[shrinkNodeIndex];
                    node->SetLeftChild(true);
                }
                // build right subtree, outer box
                uint64_t rightChildIndex = BuildMidpointSplitTreeR(box, insideBoxTo, pointsEnd, auxBeg + (insideBoxTo - pointsBeg));
                if (rightChildIndex) {
                    InnerNode* node = (InnerNode*) &_nodes[shrinkNodeIndex];
                    node->SetRightChildIndex(rightChildIndex);
                }
                return shrinkNodeIndex;
            }
        }
    }

    void RemoveTrivialNodes()
    {
        std::vector<Node> reducedNodes;
        reducedNodes.reserve(_nodes.size());
    }

    void RemoveTrivialNodesR(const Box<FloatT, Dim>& box, const Node* node, bool isParentTrivial, std::vector<Node>& reducedNodes)
    {
    }
};

#endif // AKNN_BBD_TREE_H
