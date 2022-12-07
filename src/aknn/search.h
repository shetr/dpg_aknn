#ifndef AKNN_SEARCH_H
#define AKNN_SEARCH_H

#include <vector>
#include <queue>
#include <limits>

#include "vec.h"
#include "bbd_tree.h"

template<typename FloatT, int Dim>
struct DistNode
{
    FloatT dist;
    int64_t nodeIdx;
    Box box;
};

template<typename FloatT, int Dim>
struct DistNodeCompare
{
    bool operator()(const DistNode<FloatT, Dim>& p1, const DistNode<FloatT, Dim>& p2) const {
        return !(p1.dist < p2.dist);
    }
};

template<typename FloatT, int Dim>
Vec<FloatT, Dim> NaiveFindNN(const std::vector<Vec<FloatT, Dim>>& points, const Vec<FloatT, Dim>& queryPoint)
{
    Vec<FloatT, Dim> res = points[0];
    FloatT minDist = queryPoint.DistSquared(res);
    for (int i = 1; i < (int)points.size(); ++i) {
        FloatT dist = queryPoint.DistSquared(points[i]);
        if (dist < minDist) {
            minDist = dist;
            res = points[i];
        }
    }
    return res;
}

template<typename FloatT, int Dim>
std::vector<Vec<FloatT, Dim>> NaiveFindKNN(const std::vector<Vec<FloatT, Dim>>& points, const Vec<FloatT, Dim>& queryPoint, int k)
{
    std::vector<Vec<FloatT, Dim>> res;
    return res;
}


template<typename BBDTreeType, typename PriQueueType, typename FloatT, int Dim>
Vec<FloatT, Dim> FindAproximateNearestNeighbor(const BBDTreeType& tree, const Vec<FloatT, Dim>& queryPoint, FloatT epsilon)
{
    Vec<FloatT, Dim> ann;
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
            Vec<FloatT, Dim> localNN = NaiveFindNN(node.points, queryPoint);
            FloatT localNNDist = queryPoint.DistSquared(localNN);
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
            nodeQueue.push({distRight, innerNode->GetRightChildIndex(), rightBox});
        }
    }
    return ann;
}

template<typename BBDTreeType, typename PriQueueType, typename FloatT, int Dim>
void FindKAproximateNearestNeighbors(const BBDTreeType& tree, const Vec<FloatT, Dim>& point)
{
    
}

#endif // AKNN_SEARCH_H
