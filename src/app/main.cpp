#include <iostream>
#include <array>
#include <vector>
#include <stdint.h>
#include <memory>
#include <chrono>
#include <sstream>

#include <aknn/vec.h>
#include <aknn/bbd_tree.h>
#include <aknn/search.h>

#include <argumentum/argparse-h.h>

template<int Dim>
std::vector<PointObj<float, Dim>> LoadPoints(const std::string& filename)
{
   std::vector<PointObj<float, Dim>> res;
   std::ifstream file(filename);
   if (!file) {
      std::cout << "failed to read file: " << filename << std::endl;
      return res;
   }
   Vec<float, Dim> v;
   while (file.peek() != EOF)
   {
      for (int d = 0; d < Dim; ++d) {
         file >> v[d];
      }
      res.push_back(PointObj<float, Dim>({v}));
   }
   
   return res;
}

void WritePoint(FILE* file, VecF3 point, VecF3 color)
{
   fprintf(file, "point_colored %f %f %f %f %f %f\n", point[0], point[1], point[2], color[0], color[1], color[2]);
}

void WriteSplit(FILE* file, BoxF3 box, float value, int dim, VecF3 color)
{
   VecF3 v0 = box.min;
   VecF3 v1 = box.min;
   VecF3 v2 = box.max;
   VecF3 v3 = box.max;
   v0[dim] = value;
   v1[dim] = value;
   v2[dim] = value;
   v3[dim] = value;
   int otherDim = (dim + 1) % 3;
   v1[otherDim] = box.max[otherDim];
   v3[otherDim] = box.min[otherDim];
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", v0[0], v0[1], v0[2], color[0], color[1], color[2], v1[0], v1[1], v1[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", v1[0], v1[1], v1[2], color[0], color[1], color[2], v2[0], v2[1], v2[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", v2[0], v2[1], v2[2], color[0], color[1], color[2], v3[0], v3[1], v3[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", v3[0], v3[1], v3[2], color[0], color[1], color[2], v0[0], v0[1], v0[2], color[0], color[1], color[2]);
   
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", v0[0], v0[1], v0[2], color[0], color[1], color[2], v2[0], v2[1], v2[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", v1[0], v1[1], v1[2], color[0], color[1], color[2], v3[0], v3[1], v3[2], color[0], color[1], color[2]);
}

void WriteBox(FILE* file, BoxF3 box, VecF3 color)
{
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.min[0], box.min[1], box.min[2], color[0], color[1], color[2], box.min[0], box.max[1], box.min[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.min[0], box.min[1], box.min[2], color[0], color[1], color[2], box.min[0], box.min[1], box.max[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.min[0], box.max[1], box.max[2], color[0], color[1], color[2], box.min[0], box.max[1], box.min[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.min[0], box.max[1], box.max[2], color[0], color[1], color[2], box.min[0], box.min[1], box.max[2], color[0], color[1], color[2]);
   
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.max[0], box.min[1], box.min[2], color[0], color[1], color[2], box.max[0], box.max[1], box.min[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.max[0], box.min[1], box.min[2], color[0], color[1], color[2], box.max[0], box.min[1], box.max[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.max[0], box.max[1], box.max[2], color[0], color[1], color[2], box.max[0], box.max[1], box.min[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.max[0], box.max[1], box.max[2], color[0], color[1], color[2], box.max[0], box.min[1], box.max[2], color[0], color[1], color[2]);

   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.min[0], box.min[1], box.min[2], color[0], color[1], color[2], box.max[0], box.min[1], box.min[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.min[0], box.min[1], box.max[2], color[0], color[1], color[2], box.max[0], box.min[1], box.max[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.min[0], box.max[1], box.min[2], color[0], color[1], color[2], box.max[0], box.max[1], box.min[2], color[0], color[1], color[2]);
   fprintf(file, "line_colored  %f %f %f %f %f %f %f %f %f %f %f %f\n", box.min[0], box.max[1], box.max[2], color[0], color[1], color[2], box.max[0], box.max[1], box.max[2], color[0], color[1], color[2]);
}

class QueryVizOptions : public argumentum::CommandOptions
{
private:
   std::string inputFile;
   std::string outputFile;
   int k = 1;
   float epsilon = 0;
   int leafSize = 10;
   int pointSize = 2;
   std::vector<float> queryInput;
public:
   QueryVizOptions( std::string_view name) : CommandOptions(name) {}

   void execute(const argumentum::ParseResult& res)
   {
      using namespace std::chrono;
      if (inputFile.size() > 0 && outputFile.size() > 0)
      {
         std::vector<PointObj<float, 3>> points = LoadPoints<3>(inputFile);
         BBDTree<float, 3> tree;
         Vec<float, 3> queryPoint = {queryInput[0], queryInput[1], queryInput[2]};
         // build
         {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            //tree = BBDTree<float, 3>::BuildBasicSplitTree(leafSize, points);
            tree = BBDTree<float, 3>::BuildMidpointSplitTree(leafSize, points);
            double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
            std::cout << "      build time: " << totalDuration << " s" << std::endl;
         }
         // query
         {
            HeapPriQueue<DistObj<float, 3>> priQueue;
            high_resolution_clock::time_point start = high_resolution_clock::now();
            std::vector<PointObjF3> knn = FindKAproximateNearestNeighbors<float, 3>(tree, queryPoint, k, epsilon, priQueue);
            double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
            std::cout << "aprox query time: " << totalDuration << " s" << std::endl;
         }
         // naive query
         {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            std::vector<PointObjF3> knn = LinearFindKNearestNeighbors(points, queryPoint, k);
            double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
            std::cout << "naive query time: " << totalDuration << " s" << std::endl;
         }
         // visualization
         FILE* file = fopen(outputFile.c_str(), "w");
         if (!file) {
            std::cout << "failed to write to a file " << outputFile << std::endl;
            return;
         }
         fprintf(file, "glpointsize %d\n", pointSize);
         for (const PointObj<float, 3>& point : points) {
            WritePoint(file, point.point, {1, 1, 0});
         }
         //WriteBox(file, tree.GetBBox(), {1, 1, 1});
         fprintf(file, "glpointsize %d\n", 4*pointSize);
         WritePoint(file, queryPoint, {1, 0, 0});
         HeapPriQueue<DistObj<float, 3>> priQueue;
         TraversalStats<float, 3> stats;
         std::vector<PointObjF3> knn = FindKAproximateNearestNeighbors<float, 3, Empty, true>(tree, queryPoint, k, epsilon, priQueue, stats);
         for (int i = 0; i < (int)stats.visitedNodes.size(); ++i) {
            WriteBox(file, stats.visitedNodes[i], {0, 1, 0});
         }
         for (const PointObjF3& p : knn) {
            WritePoint(file, p.point, {0, 0, 1});
         }
         fclose(file);
      }
   }
protected:
   void add_parameters(argumentum::ParameterConfig& params ) override
   {
      params.add_parameter(inputFile, "--in").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
      params.add_parameter(k, "--k").nargs(1);
      params.add_parameter(epsilon, "--eps").nargs(1);
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(pointSize, "--point_size").nargs(1);
      params.add_parameter(queryInput, "--query").nargs(3);
   }
};

class QueryStatsOptions : public argumentum::CommandOptions
{
private:
   std::string outputFile;
   int dim = 3;
   int leafSize = 10;
   int queueSelected = 0;
public:
   QueryStatsOptions(std::string_view name) : CommandOptions(name)
   {}

   void execute(const argumentum::ParseResult& res)
   {
      if (outputFile.size() > 0)
      {
         if (dim == 2) {
            Execute<2>();
         } else if (dim == 3) {
            Execute<3>();
         } else if (dim == 4) {
            Execute<4>();
         }
      }
   }
private:

   template<int Dim>
   void Execute()
   {
      using namespace std::chrono;

      std::vector<int> dataCounts = {3, 5, 7};
      std::vector<std::string> dataTypes = {"clusters", "normal", "uniform"};
      std::vector<int> ks = {1, 100};
      std::vector<float> epss = {0, 1, 10};

      int queryCount = 10;
      std::vector<Vec<float, Dim>> queryPoints;
      for (int i = 0; i < queryCount; ++i) {
         Vec<float, Dim> queryPoint;
         for (int d = 0; d < Dim; ++d) {
               queryPoint[d] = ((float)rand()) / RAND_MAX;
         }
         queryPoints.push_back(queryPoint);
      }

      std::vector<std::unique_ptr<FixedPriQueue<DistObj<float, Dim>>>> fixedQueues;
      fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<float, Dim>>>(new LinearPriQueue<DistObj<float, Dim>>()));
      fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<float, Dim>>>(new HeapPriQueue<DistObj<float, Dim>>()));
      fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<float, Dim>>>(new StdPriQueue<DistObj<float, Dim>>()));

      for (int dc = 0; dc < 3; ++dc) {
      for (int dt = 0; dt < 3; ++dt) {
         std::stringstream ss;
         ss << "data\\" << dataTypes[dt] << "_" << Dim << "d_e" << dataCounts[dc] << ".txt";
         std::string file = ss.str();

         std::vector<PointObj<float, Dim>> points = LoadPoints<Dim>(file);
         BBDTree<float, Dim> tree = BBDTree<float, Dim>::BuildMidpointSplitTree(leafSize, points);
         
         std::cout << dataTypes[dt] << " ";
         std::cout << "10^" << dataCounts[dc] << " ";
         
         TraversalStats<float, Dim> stats;
         for (int k : ks) {
         for (float eps : epss) {
            double totalTime = 0;
            double totalSteps = 0;
            for (int i = 0; i < queryCount; ++i) {
               high_resolution_clock::time_point start = high_resolution_clock::now();
               std::vector<PointObj<float, Dim>> knn = FindKAproximateNearestNeighbors<float, Dim>(tree, queryPoints[i], k, eps, *fixedQueues[queueSelected]);
               double queryDuration = duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();
               totalTime += queryDuration;
               stats.traversalSteps = 0;
               stats.visitedLeafs = 0;
               stats.visitedNodes.clear();
               knn = FindKAproximateNearestNeighbors<float, Dim, Empty, true>(tree, queryPoints[i], k, eps, *fixedQueues[queueSelected], stats);
               totalSteps += stats.traversalSteps;
            }
            double avgQueryTime = totalTime / queryCount;
            double avgTravSteps = totalSteps / queryCount;
            double naiveTime = 0;
            {
               high_resolution_clock::time_point start = high_resolution_clock::now();
               std::vector<PointObj<float, Dim>> knn = LinearFindKNearestNeighbors(points, queryPoints[0], k);
               naiveTime = duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();
            }

            // TR - traversation time
            std::cout << avgQueryTime << " ";

            // NTR - number of traversal steps
            std::cout << avgTravSteps << " ";

            // PERF
            std::cout << 1 / (avgQueryTime / 1000000) << " ";

            // S - speedup
            std::cout << naiveTime / avgQueryTime << " ";
         }
         }


         std::cout << std::endl;
      }
      }
   }

protected:
   void add_parameters(argumentum::ParameterConfig& params ) override
   {
      params.add_parameter(dim, "--dim").nargs(1);
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(queueSelected, "--queue").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
   }
};

class TreeStatsOptions : public argumentum::CommandOptions
{
private:
   std::string outputFile;
   int dim = 3;
   int leafSize = 10;
public:
   TreeStatsOptions(std::string_view name) : CommandOptions(name)
   {}

   double GetMemoryString(int memory)
   {
      return memory / (1024.0 * 1024.0);
   }

   void execute(const argumentum::ParseResult& res)
   {
      if (outputFile.size() > 0)
      {
         if (dim == 2) {
            Execute<2>();
         } else if (dim == 3) {
            Execute<3>();
         } else if (dim == 4) {
            Execute<4>();
         }
      }
   }
private:

   template<int Dim>
   void Execute()
   {
      using namespace std::chrono;
      
      //std::cout << "TB M NI NL NS NAP DMAX DAVG" << std::endl;

      std::vector<int> dataCounts = {3, 5, 7};
      std::vector<std::string> dataTypes = {"clusters", "normal", "uniform"};

      for (int dc = 0; dc < 3; ++dc) {
      for (int dt = 0; dt < 3; ++dt) {
         std::stringstream ss;
         ss << "data\\" << dataTypes[dt] << "_" << Dim << "d_e" << dataCounts[dc] << ".txt";
         std::string file = ss.str();

         std::vector<PointObj<float, Dim>> points = LoadPoints<Dim>(file);
         BBDTree<float, Dim> tree;
         double buildTime;
         {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            tree = BBDTree<float, Dim>::BuildMidpointSplitTree(leafSize, points);
            buildTime = duration_cast<std::chrono::milliseconds>(high_resolution_clock::now() - start).count();
         }

         BBDTreeStats stats = tree.GetStats();
         
         std::cout << dataTypes[dt] << " ";
         std::cout << "10^" << dataCounts[dc] << " ";

         // TB build time
         std::cout << buildTime << " ";
         // M memory consumtion [MB]
         std::cout << GetMemoryString(stats.memoryConsumption);
         // NI inner node count
         std::cout << stats.innerNodeCount << " ";
         // NL leaf node count
         std::cout << stats.leafNodeCount << " ";
         // NS shrink node percentage
         double shrinkPercentage = 100 * (double)stats.shrinkNodeCount / (double)stats.innerNodeCount;
         std::cout << shrinkPercentage << " ";
         // NAP average leaf size
         std::cout << stats.avgLeafSize << " ";
         // DMAX max depth
         std::cout << stats.maxDepth << " ";
         // DAVG average depth
         std::cout << stats.avgDepth << " ";

         std::cout << std::endl;
      }
      }
   }

protected:
   void add_parameters(argumentum::ParameterConfig& params ) override
   {
      params.add_parameter(dim, "--dim").nargs(1);
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
   }
};

class TreeVizOptions : public argumentum::CommandOptions
{
public:
   std::string inputFile;
   std::string outputFile;
   int pointSize = 1;
   int leafSize = 10;
public:
   TreeVizOptions(std::string_view name) : CommandOptions(name) {}

   void execute(const argumentum::ParseResult& res)
   {
      if (inputFile.size() > 0 && outputFile.size() > 0)
      {
         std::vector<PointObj<float, 3>> points = LoadPoints<3>(inputFile);
         BBDTree<float, 3> tree;
         //tree = BBDTree<float, 3>::BuildBasicSplitTree(leafSize, points);
         tree = BBDTree<float, 3>::BuildMidpointSplitTree(leafSize, points);

         BBDTreeStats stats = tree.GetStats();

         FILE* file = fopen(outputFile.c_str(), "w");
         if (!file) {
            std::cout << "failed to write to a file " << outputFile << std::endl;
            return;
         }

         fprintf(file, "glpointsize %d\n", pointSize);

         for (const PointObj<float, 3>& point : points) {
            WritePoint(file, point.point, {1, 1, 0});
         }

         WriteBox(file, tree.GetBBox(), {1, 1, 1});
         
         std::queue<DistNode<float, 3>> nodeQueue;
         DistNode<float, 3> rootNode{0, 0, tree.GetBBox()};
         nodeQueue.push(rootNode);
         while (!nodeQueue.empty())
         {
            DistNode<float, 3> distNode = nodeQueue.front();
            const Node* node = tree.GetNode(distNode.nodeIdx);
            nodeQueue.pop();

            float t = distNode.dist / stats.maxDepth;
            float shade = 0.5f * (1.0f - t) + t;

            if (node->GetType() != NodeType::LEAF)
            {
               const InnerNode* innerNode = (const InnerNode*)node;
               Box leftBox = distNode.box;
               Box rightBox = distNode.box;
               if (node->GetType() == NodeType::SPLIT)
               {
                  const SplitNode* splitNode = (const SplitNode*)node;
                  int splitDim = splitNode->GetSplitDim();
                  float half = (distNode.box.min[splitDim] + distNode.box.max[splitDim]) / 2;
                  leftBox.max[splitDim] = half;
                  rightBox.min[splitDim] = half;
                  WriteSplit(file, distNode.box, half, splitDim, {t, 0, t});
               }
               else if (node->GetType() == NodeType::SHRINK)
               {
                  const ShrinkNode<float, 3>* shrinkNode = (const ShrinkNode<float, 3>*)node;
                  leftBox = shrinkNode->GetShrinkBox();
                  WriteBox(file, leftBox, {0, t, 0});
               }
               float depth = distNode.dist + 1;
               if (innerNode->HasLeftChild())
                  nodeQueue.push({depth, distNode.nodeIdx + GetNodeOffset<float, 3>(node->GetType()), leftBox});
               if (innerNode->GetRightChildIndex() != 0)
                  nodeQueue.push({depth, innerNode->GetRightChildIndex(), rightBox});
            }
         }
      }
   }
protected:
   void add_parameters(argumentum::ParameterConfig& params ) override
   {
      params.add_parameter(inputFile, "--in").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(pointSize, "--point_size").nargs(1);
   }
};

class EpsGraphOptions : public argumentum::CommandOptions
{
public:
   std::string inputFile;
   std::string outputFile;
   int dim = 3;
   int k = 1;
   int leafSize = 10;
   int queryCount = 1000;
public:
   EpsGraphOptions(std::string_view name) : CommandOptions(name) {}

   void execute(const argumentum::ParseResult& res)
   {
      if (inputFile.size() > 0 && outputFile.size() > 0)
      {
         if (dim == 2) {
            Execute<2>();
         } else if (dim == 3) {
            Execute<3>();
         } else if (dim == 4) {
            Execute<4>();
         }
      }
   }
protected:
   void add_parameters(argumentum::ParameterConfig& params ) override
   {
      params.add_parameter(inputFile, "--in").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
      params.add_parameter(dim, "--dim").nargs(1);
      params.add_parameter(k, "--k").nargs(1);
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(queryCount, "--queries").nargs(1);
   }

   template<int Dim>
   void Execute()
   {
      using namespace std::chrono;
      std::vector<PointObj<float, Dim>> points = LoadPoints<Dim>(inputFile);
      BBDTree<float, Dim> tree;
      tree = BBDTree<float, Dim>::BuildMidpointSplitTree(leafSize, points);

      BBDTreeStats stats = tree.GetStats();

      FILE* file = fopen(outputFile.c_str(), "w");
      if (!file) {
         std::cout << "failed to write to a file " << outputFile << std::endl;
         return;
      }
      
      std::vector<Vec<float, Dim>> queryPoints;
      for (int i = 0; i < queryCount; ++i) {
         Vec<float, Dim> queryPoint;
         for (int d = 0; d < Dim; ++d) {
               queryPoint[d] = ((float)rand()) / RAND_MAX;
         }
         queryPoints.push_back(queryPoint);
      }

      for (int i_eps = 0; i_eps <= 100; ++i_eps)
      {
         float epsilon = i_eps * 0.1f;
         fprintf(file, "%f ", epsilon);
      }
      fprintf(file, "\n");

      for (int i_eps = 0; i_eps <= 100; ++i_eps)
      {
         float epsilon = i_eps * 0.1f;

         double totalTime = 0;
         for (int i = 0; i < queryCount; ++i) {
            HeapPriQueue<DistObj<float, Dim>> priQueue;
            high_resolution_clock::time_point start = high_resolution_clock::now();
            std::vector<PointObj<float, Dim>> knn = FindKAproximateNearestNeighbors<float, Dim>(tree, queryPoints[i], k, epsilon, priQueue);
            double queryDuration = duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();
            totalTime += queryDuration;
         }
         double avgQueryTime = totalTime / queryCount;
         fprintf(file, "%f ", avgQueryTime);
      }
      fprintf(file, "\n");

      fclose(file);
   }
};

class QueueGraphOptions : public argumentum::CommandOptions
{
public:
   std::string inputFile;
   std::string outputFile;
   int dim = 3;
   float epsilon = 0;
   int leafSize = 10;
public:
   QueueGraphOptions(std::string_view name) : CommandOptions(name) {}

   void execute(const argumentum::ParseResult& res)
   {
      if (inputFile.size() > 0 && outputFile.size() > 0)
      {
         if (dim == 2) {
            Execute<2>();
         } else if (dim == 3) {
            Execute<3>();
         } else if (dim == 4) {
            Execute<4>();
         }
      }
   }
protected:
   void add_parameters(argumentum::ParameterConfig& params ) override
   {
      params.add_parameter(inputFile, "--in").nargs(1);
      params.add_parameter(dim, "--dim").nargs(1);
      params.add_parameter(epsilon, "--eps").nargs(1);
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
   }

   template<int Dim>
   void Execute()
   {
      using namespace std::chrono;
      std::vector<PointObj<float, Dim>> points = LoadPoints<Dim>(inputFile);
      BBDTree<float, Dim> tree;
      tree = BBDTree<float, Dim>::BuildMidpointSplitTree(leafSize, points);

      BBDTreeStats stats = tree.GetStats();

      FILE* file = fopen(outputFile.c_str(), "w");
      if (!file) {
         std::cout << "failed to write to a file " << outputFile << std::endl;
         return;
      }
      
      int queryCount = 1000;
      std::vector<Vec<float, Dim>> queryPoints;
      for (int i = 0; i < queryCount; ++i) {
         Vec<float, Dim> queryPoint;
         for (int d = 0; d < Dim; ++d) {
               queryPoint[d] = ((float)rand()) / RAND_MAX;
         }
         queryPoints.push_back(queryPoint);
      }

      std::vector<std::unique_ptr<FixedPriQueue<DistObj<float, Dim>>>> fixedQueues;
      fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<float, Dim>>>(new LinearPriQueue<DistObj<float, Dim>>()));
      fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<float, Dim>>>(new HeapPriQueue<DistObj<float, Dim>>()));
      fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<float, Dim>>>(new StdPriQueue<DistObj<float, Dim>>()));

      for (int q = -1; q < 3; ++q)
      {
         for (int k = 1; k <= 1200; k += std::max(1, k / 4))
         {
            if (q == -1) {
               fprintf(file, "%f ", (double)k);
            } else {
               double totalTime = 0;
               for (int i = 0; i < queryCount; ++i) {
                  high_resolution_clock::time_point start = high_resolution_clock::now();
                  std::vector<PointObj<float, Dim>> knn = FindKAproximateNearestNeighbors<float, Dim>(tree, queryPoints[i], k, epsilon, *fixedQueues[q]);
                  double queryDuration = duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();
                  totalTime += queryDuration;
               }
               double avgQueryTime = totalTime / queryCount;
               fprintf(file, "%f ", avgQueryTime);
            }
         }
         fprintf(file, "\n");
      }


      fclose(file);
   }
};

int main(int argc, char** argv)
{
   using namespace argumentum;

   argument_parser parser;
   ParameterConfig params = parser.params();
   parser.config().program( argv[0] ).description( "Aproximate k nearest neighbor search" );

   std::shared_ptr<TreeStatsOptions> treeStatsOptions = std::make_shared<TreeStatsOptions>("tree_stats");
   std::shared_ptr<QueryStatsOptions> queryStatsOptions = std::make_shared<QueryStatsOptions>("query_stats");
   std::shared_ptr<TreeVizOptions> treeVizOptions = std::make_shared<TreeVizOptions>("tree_viz");
   std::shared_ptr<QueryVizOptions> queryVizOptions = std::make_shared<QueryVizOptions>("query_viz");
   std::shared_ptr<EpsGraphOptions> epsGraphOptions = std::make_shared<EpsGraphOptions>("eps_graph");
   std::shared_ptr<QueueGraphOptions> queueGraphOptions = std::make_shared<QueueGraphOptions>("queue_graph");

   params.add_command(treeStatsOptions).help("Tree statistics.");
   params.add_command(queryStatsOptions).help("Query statistics.");
   params.add_command(treeVizOptions).help("Visualize tree hierarchy.");
   params.add_command(queryVizOptions).help("Visualize specified query.");
   params.add_command(epsGraphOptions).help("Dependence of execution time on epsilon.");
   params.add_command(queueGraphOptions).help("Dependence of execution time on queue type and k.");

   ParseResult res = parser.parse_args( argc, argv, 1 );
   if ( !res )
      return 1;

   auto pcmd = res.commands.back();
   if ( !pcmd )
      return 1;

   pcmd->execute( res );

   return 0;
}