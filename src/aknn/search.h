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


template<int Dim>
using FindNNFunc = std::function<PointObj<double, Dim>(const std::vector<PointObj<double, Dim>> objs, const Vec<double, Dim>& queryPoint)>;

template<int Dim>
using FindKNNFunc = std::function<std::vector<PointObj<double, Dim>>(const std::vector<PointObj<double, Dim>> objs, const Vec<double, Dim>& queryPoint, int k)>;

template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim> LinearFindNearestNeighborInRange(const PointObj<FloatT, Dim, ObjData>* objsBeg, const PointObj<FloatT, Dim, ObjData>* objsEnd, const Vec<FloatT, Dim>& queryPoint)
{
    PointObj<FloatT, Dim, ObjData> res = *objsBeg;
    FloatT minDist = queryPoint.DistSquared(res.point);
    for (const PointObj<FloatT, Dim, ObjData>* objIt = objsBeg + 1; objIt != objsEnd; ++objIt) {
        FloatT dist = queryPoint.DistSquared(objIt->point);
        if (dist < minDist) {
            minDist = dist;
            res = *objIt;
        }
    }
    return res;
}

template<typename FloatT, int Dim, typename ObjData = Empty>
PointObj<FloatT, Dim> LinearFindNearestNeighbor(const std::vector<PointObj<FloatT, Dim, ObjData>>& objs, const Vec<FloatT, Dim>& queryPoint)
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
            PointObj<FloatT, Dim, ObjData> localNN = LinearFindNearestNeighborInRange<FloatT, Dim, ObjData>(
                tree.GetObj(leafNode->GetPointsBegIndex()), tree.GetObj(leafNode->GetPointsEndIndex()), queryPoint);
            FloatT localNNDist = queryPoint.DistSquared(localNN.point);
            if (localNNDist < minDist)
            {
                minDist = localNNDist;
                ann = localNN;
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
                const ShrinkNode<FloatT, Dim>* splitNode = (const ShrinkNode<FloatT, Dim>*)node;
                rightBox = splitNode->GetShrinkBox();
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
std::vector<PointObj<FloatT, Dim, ObjData>> FindKAproximateNearestNeighbors(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, int k, FloatT epsilon)
{
    std::vector<PointObj<FloatT, Dim, ObjData>> res;

    return res;
}

template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<PointObj<FloatT, Dim, ObjData>> FindKNearestNeighbors(const BBDTree<FloatT, Dim, ObjData>& tree, const Vec<FloatT, Dim>& queryPoint, int k)
{
    return FindKAproximateNearestNeighbors<FloatT, Dim, ObjData>(tree, queryPoint, k, 0);
}

#endif // AKNN_SEARCH_H
