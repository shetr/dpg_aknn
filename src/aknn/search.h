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
    int64_t nodeIdx;
    Box<FloatT, Dim> box;
};

template<typename FloatT, int Dim>
struct DistObj
{
    FloatT dist;
    PointObj<FloatT, Dim> obj;
};

template<typename FloatT, int Dim>
struct DistNodeCompare
{
    bool operator()(const DistNode<FloatT, Dim>& p1, const DistNode<FloatT, Dim>& p2) const {
        return !(p1.dist < p2.dist);
    }
};

template<typename FloatT, int Dim>
struct DistObjCompare
{
    bool operator()(const DistObj<FloatT, Dim>& p1, const DistObj<FloatT, Dim>& p2) const {
        return p1.dist < p2.dist;
    }
};

template<typename FloatT, int Dim>
PointObj<FloatT, Dim> LinearFindNN(const std::vector<PointObj<FloatT, Dim>>& objs, const Vec<FloatT, Dim>& queryPoint)
{
    PointObj<FloatT, Dim> res = objs[0];
    FloatT minDist = queryPoint.DistSquared(res.point);
    for (int i = 1; i < (int)objs.size(); ++i) {
        FloatT dist = queryPoint.DistSquared(objs[i].point);
        if (dist < minDist) {
            minDist = dist;
            res = objs[i];
        }
    }
    return res;
}

template<typename FloatT, int Dim>
std::vector<PointObj<FloatT, Dim>> LinearFindKNN(const std::vector<PointObj<FloatT, Dim>>& objs, const Vec<FloatT, Dim>& queryPoint, int k)
{
    LinearPriQueue<DistObj<FloatT, Dim>> priQueue;
    priQueue.Init(k, DistObjCompare<FloatT, Dim>());
    for (int i = 0; i < (int)objs.size(); ++i) {
        priQueue.Push({queryPoint.DistSquared(objs[i].point), objs[i]});
    }
    std::vector<DistObj<FloatT, Dim>> values = priQueue.GetValues();
    std::vector<PointObj<FloatT, Dim>> res;
    res.reserve(values.size());
    for (int i = 0; i < (int)values.size(); ++i) {
        res.push_back(values[i].obj);
    }
    return res;
}


template<typename FloatT, int Dim>
PointObj<FloatT, Dim> FindAproximateNearestNeighbor(const BBDTree<FloatT, Dim>& tree, const Vec<FloatT, Dim>& queryPoint, FloatT epsilon)
{
    PointObj<FloatT, Dim> ann;
    FloatT minDist = std::numeric_limits<FloatT>::infinity();
    std::priority_queue<DistNode> nodeQueue;
    DistNode rootNode{0, 0, tree.GetBBox()};
    nodeQueue.push(rootNode);
    while (!nodeQueue.empty())
    {
        DistNode distNode = nodeQueue.top();
        Node* node = tree.GetNode(distNode.nodeIdx);
        nodeQueue.pop();
        
        if (node->GetType() == NodeType::LEAF)
        {
            PointObj<FloatT, Dim> localNN = LinearFindNN(node.points, queryPoint);
            FloatT localNNDist = queryPoint.DistSquared(localNN.point);
            if (localNNDist < minDist)
            {
                minDist = localNNDist;
                ann = localNN;
            }
            else if (localNNDist > minDist / (1 + epsilon))
            {
                break;
            }
        }
        else
        {
            InnerNode* innerNode = (InnerNode*)node;
            Box leftBox = distNode.box;
            Box rightBox = distNode.box;
            if (node->GetType() == NodeType::SPLIT)
            {
                SplitNode* splitNode = (SplitNode*)node;
                int splitDim = splitNode->GetSplitDim();
                FloatT half = (distNode.box.min[splitDim] + distNode.box.max[splitDim]) / 2;
                leftBox.max[splitDim] = half;
                rightBox.min[splitDim] = half;
            }
            else if (node->GetType() == NodeType::SHRINK)
            {
                ShrinkNode* splitNode = (ShrinkNode*)node;
                rightBox = splitNode->GetShrinkBox();
            }
            FloatT distLeft = leftBox.SquaredDistance(queryPoint);
            FloatT distRight = rightBox.SquaredDistance(queryPoint);
            if (innerNode->HasLeftChild())
                nodeQueue.push({distLeft, distNode.nodeIdx + GetNodeOffset(node->GetType()), leftBox});
            if (innerNode->GetRightChildIndex() != 0)
                nodeQueue.push({distRight, innerNode->GetRightChildIndex(), rightBox});
        }
    }
    return ann;
}

template<typename FloatT, int Dim>
std::vector<PointObj<FloatT, Dim>> FindKAproximateNearestNeighbors(const BBDTree<FloatT, Dim>& tree, const Vec<FloatT, Dim>& queryPoint, int k)
{
    
}

#endif // AKNN_SEARCH_H
