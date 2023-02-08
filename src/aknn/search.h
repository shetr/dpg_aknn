#ifndef AKNN_SEARCH_H
#define AKNN_SEARCH_H

#include <vector>
#include <queue>
#include <limits>

#include "vec.h"
#include "bbd_tree.h"
#include "pri_queue.h"

//! BBD tree node with additional information of distance to query point + bounding box 
template<typename FloatT, int Dim>
struct DistNode
{
    //! Distance from query point to this node
    FloatT dist;
    //! Index of this node
    index_t nodeIdx;
    //! bounding box of this node
    Box<FloatT, Dim> box;
};

//! Point object with aditional information of distance to some query point
template<typename FloatT, int Dim, typename ObjData = Empty>
struct DistObj
{
    //! distance of this object from query point
    FloatT dist;
    //! point object
    PointObj<FloatT, Dim, ObjData> obj;
};

//! Converts DistObj array to PointObj array
template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim, ObjData>> DistObjsToPointObjs(const std::vector<DistObj<FloatT, Dim, ObjData>>& distObjs) {
    std::vector<PointObj<FloatT, Dim, ObjData>> res;
    res.resize(distObjs.size());
    std::transform(distObjs.begin(), distObjs.end(), res.begin(), [&](const DistObj<FloatT, Dim, ObjData>& distObj) { return distObj.obj; });
    return res;
}

//! Comparator for DistNode objects
template<typename FloatT, int Dim>
struct DistNodeCompare
{
    bool operator()(const DistNode<FloatT, Dim>& p1, const DistNode<FloatT, Dim>& p2) const {
        return !(p1.dist <= p2.dist);
    }
};
//! Comparator for DistObj objects
template<typename FloatT, int Dim, typename ObjData = Empty>
struct DistObjCompare
{
    bool operator()(const DistObj<FloatT, Dim, ObjData>& p1, const DistObj<FloatT, Dim, ObjData>& p2) const {
        return p1.dist < p2.dist;
    }
};

//! Naive linear nearest neighbor in specified range with DistObj result
template<typename FloatT, int Dim, typename ObjData = Empty>
DistObj<FloatT, Dim, ObjData> LinearFindNearestNeighborInRangeWithDist(const PointObj<FloatT, Dim, ObjData>* objsBeg, const PointObj<FloatT, Dim, ObjData>* objsEnd, const Vec<FloatT, Dim>& queryPoint)
{
    PointObj<FloatT, Dim, ObjData> nnObj = *objsBeg;
    FloatT minDist = queryPoint.DistSquared(nnObj.point);
    for (const PointObj<FloatT, Dim, ObjData>* objIt = objsBeg + 1; objIt != objsEnd; ++objIt) {
        FloatT dist = queryPoint.DistSquared(objIt->point);
        if (dist < minDist) {
            minDist = dist;
            nnObj = *objIt;
        }
    }
    return { minDist, nnObj };
}
//! Naive linear nearest neighbor in specified range
template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData> LinearFindNearestNeighborInRange(const PointObj<FloatT, Dim, ObjData>* objsBeg, const PointObj<FloatT, Dim, ObjData>* objsEnd, const Vec<FloatT, Dim>& queryPoint)
{
    return LinearFindNearestNeighborInRangeWithDist<FloatT, Dim, ObjData>(objsBeg, objsEnd, queryPoint).obj;
}
//! Naive linear nearest neighbor
template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData> LinearFindNearestNeighbor(const std::vector<PointObj<FloatT, Dim, ObjData>>& objs, const Vec<FloatT, Dim>& queryPoint)
{
    return LinearFindNearestNeighborInRange<FloatT, Dim, ObjData>(objs.data(), objs.data() + objs.size(), queryPoint);
}
//! Naive linear k nearest neighbors
template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim, ObjData>> LinearFindKNearestNeighbors(const std::vector<PointObj<FloatT, Dim, ObjData>>& objs, const Vec<FloatT, Dim>& queryPoint, int k)
{
    LinearPriQueue<DistObj<FloatT, Dim, ObjData>> priQueue;
    priQueue.Init(k, DistObjCompare<FloatT, Dim, ObjData>());
    for (int i = 0; i < (int)objs.size(); ++i) {
        priQueue.Push({queryPoint.DistSquared(objs[i].point), objs[i]});
    }
    return DistObjsToPointObjs(priQueue.GetValues());
}

//! Statistics of FindAproximateNearestNeighbor and FindKAproximateNearestNeighbors
template<typename FloatT, int Dim>
struct TraversalStats
{
    int traversalSteps = 0;
    int visitedLeafs = 0;
    std::vector<Box<FloatT, Dim>> visitedNodes;
};

//! Priority queue used for nodes inside FindAproximateNearestNeighbor and FindKAproximateNearestNeighbors
template<typename FloatT, int Dim>
using DistNodePriQueue = std::priority_queue<DistNode<FloatT, Dim>, std::vector<DistNode<FloatT, Dim>>, DistNodeCompare<FloatT, Dim>>;

//! Push child nodes of some node to node priority queue. Used inside FindAproximateNearestNeighbor and FindKAproximateNearestNeighbors.
template<typename FloatT, int Dim, typename ObjData>
void PushChildsToNodeQueue(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, const DistNode<FloatT, Dim>& distNode, DistNodePriQueue<FloatT, Dim>& nodeQueue)
{
    const Node* node = tree.GetNode(distNode.nodeIdx);
    const InnerNode* innerNode = (const InnerNode*)node;
    Box leftBox = distNode.box;
    Box rightBox = distNode.box;
    FloatT distLeft;
    FloatT distRight;
    if (node->GetType() == NodeType::SPLIT)
    {
        const SplitNode* splitNode = (const SplitNode*)node;
        int splitDim = splitNode->GetSplitDim();
        FloatT half = (distNode.box.min[splitDim] + distNode.box.max[splitDim]) / 2;
        leftBox.max[splitDim] = half;
        rightBox.min[splitDim] = half;
        
        distLeft = leftBox.SquaredDistance(queryPoint);
        distRight = rightBox.SquaredDistance(queryPoint);
    }
    else if (node->GetType() == NodeType::SHRINK)
    {
        const ShrinkNode<FloatT, Dim>* shrinkNode = (const ShrinkNode<FloatT, Dim>*)node;
        leftBox = shrinkNode->GetShrinkBox();
        
        distLeft = leftBox.SquaredDistance(queryPoint);
        distRight = distNode.dist;
    }
    if (innerNode->HasLeftChild())
        nodeQueue.push({distLeft, distNode.nodeIdx + GetNodeOffset<FloatT, Dim>(node->GetType()), leftBox});
    if (innerNode->GetRightChildIndex() != 0)
        nodeQueue.push({distRight, innerNode->GetRightChildIndex(), rightBox});
}

//! Finds aproximate nearest neighbor using BBD tree
template<typename FloatT, int Dim, typename ObjData = Empty, bool measureStats = false>
PointObj<FloatT, Dim, ObjData> FindAproximateNearestNeighbor(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, FloatT epsilon, TraversalStats<FloatT, Dim>& stats)
{
    PointObj<FloatT, Dim, ObjData> ann;
    FloatT minDist = std::numeric_limits<FloatT>::infinity();
    DistNodePriQueue<FloatT, Dim> nodeQueue;
    DistNode<FloatT, Dim> rootNode{tree.GetBBox().SquaredDistance(queryPoint), 0, tree.GetBBox()};
    nodeQueue.push(rootNode);
    while (!nodeQueue.empty())
    {
        DistNode<FloatT, Dim> distNode = nodeQueue.top();
        const Node* node = tree.GetNode(distNode.nodeIdx);
        nodeQueue.pop();

        if (measureStats)
        {
            ++stats.traversalSteps;
            stats.visitedNodes.push_back(distNode.box);
        }

        if (distNode.dist > minDist / (1 + epsilon)) {
            break;
        }
        
        if (node->GetType() == NodeType::LEAF)
        {
            const LeafNode* leafNode = (const LeafNode*)node;
            DistObj<FloatT, Dim, ObjData> localNN = LinearFindNearestNeighborInRangeWithDist<FloatT, Dim, ObjData>(
                tree.GetObj(leafNode->GetPointsBegIndex()), tree.GetObj(leafNode->GetPointsEndIndex()), queryPoint);
            if (localNN.dist < minDist)
            {
                minDist = localNN.dist;
                ann = localNN.obj;
            }

            if (measureStats)
                ++stats.visitedLeafs;
        }
        else
        {
            PushChildsToNodeQueue(tree, queryPoint, distNode, nodeQueue);
        }
    }
    return ann;
}
//! Finds aproximate nearest neighbor using BBD tree
template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData> FindAproximateNearestNeighbor(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, FloatT epsilon)
{
    TraversalStats<FloatT, Dim> dummyStats;
    return FindAproximateNearestNeighbor<FloatT, Dim, ObjData>(tree, queryPoint, epsilon, dummyStats);
}
//! Finds nearest neighbor using BBD tree
template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData> FindNearestNeighbor(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint)
{
    return FindAproximateNearestNeighbor<FloatT, Dim, ObjData>(tree, queryPoint, 0);
}

//! Finds k aproximate nearest neighbors using BBD tree
template<typename FloatT, int Dim, typename ObjData = Empty, bool measureStats = false>
std::vector<PointObj<FloatT, Dim, ObjData>> FindKAproximateNearestNeighbors(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, int k, FloatT epsilon, FixedPriQueue<DistObj<FloatT, Dim, ObjData>>& aknnQueue, TraversalStats<FloatT, Dim>& stats)
{
    if (k == 1) {
        return { FindAproximateNearestNeighbor<FloatT, Dim, ObjData, measureStats>(tree, queryPoint, epsilon, stats) };
    }

    DistObjCompare<FloatT, Dim, ObjData> distObjCompare;
    aknnQueue.Init(k, distObjCompare);

    DistNodePriQueue<FloatT, Dim> nodeQueue;
    DistNode<FloatT, Dim> rootNode{tree.GetBBox().SquaredDistance(queryPoint), 0, tree.GetBBox()};
    nodeQueue.push(rootNode);
    while (!nodeQueue.empty())
    {
        DistNode<FloatT, Dim> distNode = nodeQueue.top();
        const Node* node = tree.GetNode(distNode.nodeIdx);
        nodeQueue.pop();
        
        if (measureStats)
        {
            ++stats.traversalSteps;
            stats.visitedNodes.push_back(distNode.box);
        }

        if (aknnQueue.IsFull() && distNode.dist > aknnQueue.GetLast().dist / (1 + epsilon)) {
            break;
        }
        
        if (node->GetType() == NodeType::LEAF)
        {
            const LeafNode* leafNode = (const LeafNode*)node;
            const PointObj<FloatT, Dim, ObjData>* leafBeg = tree.GetObj(leafNode->GetPointsBegIndex());
            const PointObj<FloatT, Dim, ObjData>* leafEnd = tree.GetObj(leafNode->GetPointsEndIndex());
            for (const PointObj<FloatT, Dim, ObjData>* objPtr = leafBeg; objPtr != leafEnd; ++objPtr) {
                aknnQueue.Push(DistObj<FloatT, Dim, ObjData>({queryPoint.DistSquared(objPtr->point), *objPtr}));
            }
            
            if (measureStats)
                ++stats.visitedLeafs;
        }
        else
        {
            PushChildsToNodeQueue(tree, queryPoint, distNode, nodeQueue);
        }
    }

    return DistObjsToPointObjs(aknnQueue.GetValues());
}
//! Finds k aproximate nearest neighbors using BBD tree
template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim, ObjData>> FindKAproximateNearestNeighbors(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, int k, FloatT epsilon, FixedPriQueue<DistObj<FloatT, Dim, ObjData>>& aknnQueue)
{
    TraversalStats<FloatT, Dim> dummyStats;
    return FindKAproximateNearestNeighbors<FloatT, Dim, ObjData, false>(tree, queryPoint, k, epsilon, aknnQueue, dummyStats);
}
//! Finds k nearest neighbors using BBD tree
template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim, ObjData>> FindKNearestNeighbors(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, int k, FixedPriQueue<DistObj<FloatT, Dim, ObjData>>& knnQueue)
{
    return FindKAproximateNearestNeighbors<FloatT, Dim, ObjData>(tree, queryPoint, k, 0, knnQueue);
}

#endif // AKNN_SEARCH_H
