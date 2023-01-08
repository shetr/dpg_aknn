#ifndef AKNN_SEARCH_H
#define AKNN_SEARCH_H

#include <vector>
#include <queue>
#include <limits>

#include "vec.h"
#include "bbd_tree.h"
#include "pri_queue.h"

template<typename FloatT, int Dim>
struct DistNode
{
    FloatT dist;
    uint64_t nodeIdx;
    Box<FloatT, Dim> box;
};

template<typename FloatT, int Dim, typename ObjData = Empty>
struct DistObj
{
    FloatT dist;
    PointObj<FloatT, Dim, ObjData> obj;
};

template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim, ObjData>> DistObjsToPointObjs(const std::vector<DistObj<FloatT, Dim, ObjData>>& distObjs) {
    std::vector<PointObj<FloatT, Dim, ObjData>> res;
    res.resize(distObjs.size());
    std::transform(distObjs.begin(), distObjs.end(), res.begin(), [&](const DistObj<FloatT, Dim, ObjData>& distObj) { return distObj.obj; });
    return res;
}

template<typename FloatT, int Dim>
struct DistNodeCompare
{
    bool operator()(const DistNode<FloatT, Dim>& p1, const DistNode<FloatT, Dim>& p2) const {
        return !(p1.dist <= p2.dist);
    }
};

template<typename FloatT, int Dim, typename ObjData = Empty>
struct DistObjCompare
{
    bool operator()(const DistObj<FloatT, Dim, ObjData>& p1, const DistObj<FloatT, Dim, ObjData>& p2) const {
        return p1.dist < p2.dist;
    }
};

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

template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData> LinearFindNearestNeighborInRange(const PointObj<FloatT, Dim, ObjData>* objsBeg, const PointObj<FloatT, Dim, ObjData>* objsEnd, const Vec<FloatT, Dim>& queryPoint)
{
    return LinearFindNearestNeighborInRangeWithDist<FloatT, Dim, ObjData>(objsBeg, objsEnd, queryPoint).obj;
}

template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData> LinearFindNearestNeighbor(const std::vector<PointObj<FloatT, Dim, ObjData>>& objs, const Vec<FloatT, Dim>& queryPoint)
{
    return LinearFindNearestNeighborInRange<FloatT, Dim, ObjData>(objs.data(), objs.data() + objs.size(), queryPoint);
}

template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim>> LinearFindKNearestNeighbors(const std::vector<PointObj<FloatT, Dim, ObjData>>& objs, const Vec<FloatT, Dim>& queryPoint, int k)
{
    LinearPriQueue<DistObj<FloatT, Dim, ObjData>> priQueue;
    priQueue.Init(k, DistObjCompare<FloatT, Dim, ObjData>());
    for (int i = 0; i < (int)objs.size(); ++i) {
        priQueue.Push({queryPoint.DistSquared(objs[i].point), objs[i]});
    }
    return DistObjsToPointObjs(priQueue.GetValues());
}


template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData> FindAproximateNearestNeighbor(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, FloatT epsilon)
{
    // TODO: handle edge case when there are 0 nodes
    PointObj<FloatT, Dim, ObjData> ann;
    FloatT minDist = std::numeric_limits<FloatT>::infinity();
    std::priority_queue<DistNode<FloatT, Dim>, std::vector<DistNode<FloatT, Dim>>, DistNodeCompare<FloatT, Dim>> nodeQueue;
    DistNode<FloatT, Dim> rootNode{0, 0, tree.GetBBox()};
    nodeQueue.push(rootNode);
    while (!nodeQueue.empty())
    {
        DistNode<FloatT, Dim> distNode = nodeQueue.top();
        const Node* node = tree.GetNode(distNode.nodeIdx);
        nodeQueue.pop();

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
        }
        else
        {
            const InnerNode* innerNode = (const InnerNode*)node;
            Box leftBox = distNode.box;
            Box rightBox = distNode.box;
            if (node->GetType() == NodeType::SPLIT)
            {
                const SplitNode* splitNode = (const SplitNode*)node;
                int splitDim = splitNode->GetSplitDim();
                FloatT half = (distNode.box.min[splitDim] + distNode.box.max[splitDim]) / 2;
                leftBox.max[splitDim] = half;
                rightBox.min[splitDim] = half;
            }
            else if (node->GetType() == NodeType::SHRINK)
            {
                const ShrinkNode<FloatT, Dim>* shrinkNode = (const ShrinkNode<FloatT, Dim>*)node;
                leftBox = shrinkNode->GetShrinkBox();
            }
            FloatT distLeft = leftBox.SquaredDistance(queryPoint);
            FloatT distRight = rightBox.SquaredDistance(queryPoint);
            if (innerNode->HasLeftChild())
                nodeQueue.push({distLeft, distNode.nodeIdx + GetNodeOffset<FloatT, Dim>(node->GetType()), leftBox});
            if (innerNode->GetRightChildIndex() != 0)
                nodeQueue.push({distRight, innerNode->GetRightChildIndex(), rightBox});
        }
    }
    return ann;
}
template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim, ObjData> FindNearestNeighbor(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint)
{
    return FindAproximateNearestNeighbor<FloatT, Dim, ObjData>(tree, queryPoint, 0);
}

template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim, ObjData>> FindKAproximateNearestNeighbors(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, int k, FloatT epsilon, FixedPriQueue<DistObj<FloatT, Dim, ObjData>>& aknnQueue)
{
    // TODO: handle edge case when there are 0 nodes
    DistObjCompare<FloatT, Dim, ObjData> distObjCompare;
    aknnQueue.Init(k, distObjCompare);

    std::priority_queue<DistNode<FloatT, Dim>, std::vector<DistNode<FloatT, Dim>>, DistNodeCompare<FloatT, Dim>> nodeQueue;
    DistNode<FloatT, Dim> rootNode{0, 0, tree.GetBBox()};
    nodeQueue.push(rootNode);
    while (!nodeQueue.empty())
    {
        DistNode<FloatT, Dim> distNode = nodeQueue.top();
        const Node* node = tree.GetNode(distNode.nodeIdx);
        nodeQueue.pop();

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
        }
        else
        {
            // TODO: generalize for both nn and knn
            const InnerNode* innerNode = (const InnerNode*)node;
            Box leftBox = distNode.box;
            Box rightBox = distNode.box;
            if (node->GetType() == NodeType::SPLIT)
            {
                const SplitNode* splitNode = (const SplitNode*)node;
                int splitDim = splitNode->GetSplitDim();
                FloatT half = (distNode.box.min[splitDim] + distNode.box.max[splitDim]) / 2;
                leftBox.max[splitDim] = half;
                rightBox.min[splitDim] = half;
            }
            else if (node->GetType() == NodeType::SHRINK)
            {
                const ShrinkNode<FloatT, Dim>* shrinkNode = (const ShrinkNode<FloatT, Dim>*)node;
                leftBox = shrinkNode->GetShrinkBox();
            }
            FloatT distLeft = leftBox.SquaredDistance(queryPoint);
            FloatT distRight = rightBox.SquaredDistance(queryPoint);
            if (innerNode->HasLeftChild())
                nodeQueue.push({distLeft, distNode.nodeIdx + GetNodeOffset<FloatT, Dim>(node->GetType()), leftBox});
            if (innerNode->GetRightChildIndex() != 0)
                nodeQueue.push({distRight, innerNode->GetRightChildIndex(), rightBox});
        }
    }

    return DistObjsToPointObjs(aknnQueue.GetValues());
}

template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim, ObjData>> FindKNearestNeighbors(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, int k, FixedPriQueue<DistObj<FloatT, Dim, ObjData>>& knnQueue)
{
    return FindKAproximateNearestNeighbors<FloatT, Dim, ObjData>(tree, queryPoint, k, 0, knnQueue);
}

#endif // AKNN_SEARCH_H
