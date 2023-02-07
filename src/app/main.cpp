#include <iostream>
#include <array>
#include <vector>
#include <stdint.h>

// udelat nejdrive uplne naivni nn a knn pro jednoduchost (linearni pruchod)
// TODO: v hlavickach, pokud mam implementaci funkci, tak ji presunout na konec souboru
// TODO: mozna presunout search funkce BBD stromu do jeho tride, ale to jen za predpokladu ze nechci delat dalsi BBD strom
// TODO: nejdriv napsat fair split strom (FairSplitTree)
// TODO: udelat strukturu testovani a zpusob cneni vstupnich dat, zobrazovani vystupnich dat
// TODO: napsat naivni BBD strom (NaiveBBDTree)
// TODO: napsat prostorove optimalni BBD strom (TinyBBDTree)
// TODO: unit testy - hlavne na fronty, ale dali by se pouzit i na jednoduche testy NN, BBD atd ...

// plan:
// 2 binarky: jedna knihovna s aknn, testovaci aplikace
// aknn obsahuje ciste kod k aknn
// testovaci aplikace ma argumenty pro urceni typu behu - mereni, nebo vystup do textoveho formatu, nebo mozna bych mohl udelat vystup do obrazku
// mozna argumenty pro nastaveni dimenze, ale to by mohlo jit poznat ze souboru - vymyslet format souboru
// - asi bych udelal jednotny, bud textovy, nebo binarni, nejspis by prvni 2 polzky byli pocet bodu a jejich dimenze
// mozna parametry na vyber vlastnosti, treba typ prioritni fronty, nebo epsilon, pocet prvku v listech, v situaci ze delam konkretni beh
// - tim padem jeste treti typ aplikace na jednoduchy test pro konkretni parametry
// - nebo prijimat konfiguracni soubor jake parametry merit
// takze treba volani:
// testapp measure data.txt > measured.txt
// testapp measure --cfg config.txt > measured.txt
// testapp gen_viz data.txt > viz.txt
// - pokud to pujde udelat zobrazovani samotneho stromu, nebo i vizualizaci k nejblizsich sousedu vuci nejakemu bodu

// asi si sem pridam i python cast na generovani grafu apod

// tri prikazy: measure, stats, visualize

#include <aknn/vec.h>
#include <aknn/bbd_tree.h>
#include <aknn/search.h>

#include <argumentum/argparse-h.h>

#include <memory>
#include <chrono>

class GlobalOptions : public argumentum::Options
{
public:
   std::string inputFile;
   std::string outputFile;
   int dim = 3;
   int k = 1;
   float epsilon = 0;
   int leafSize = 10;
   void add_parameters(argumentum::ParameterConfig& params) override
   {
      params.add_parameter(inputFile, "--in").nargs(1);
      params.add_parameter(dim, "--dim").nargs(1);
      params.add_parameter(k, "--k").nargs(1);
      params.add_parameter(epsilon, "--eps").nargs(1);
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
   }
};

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

class MeasureOptions : public argumentum::CommandOptions
{
   std::shared_ptr<GlobalOptions> _globalOptions;
public:
   MeasureOptions( std::string_view name, std::shared_ptr<GlobalOptions> globalOptions)
      : CommandOptions(name), _globalOptions(globalOptions)
   {}

   void execute(const argumentum::ParseResult& res)
   {
      using namespace std::chrono;
      if (_globalOptions && _globalOptions->inputFile.size() > 0 )
      {
         std::vector<PointObj<float, 3>> points = LoadPoints<3>(_globalOptions->inputFile);
         BBDTree<float, 3> tree;
         // build
         {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            //tree = BBDTree<float, 3>::BuildBasicSplitTree(_globalOptions->leafSize, points);
            tree = BBDTree<float, 3>::BuildMidpointSplitTree(_globalOptions->leafSize, points);
            double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
            std::cout << "      build time: " << totalDuration << " s" << std::endl;
         }
         Vec<float, 3> queryPoint = {0, 0, 0};
         // query
         {
            HeapPriQueue<DistObj<float, 3>> priQueue;
            high_resolution_clock::time_point start = high_resolution_clock::now();
            std::vector<PointObjF3> knn = FindKAproximateNearestNeighbors<float, 3>(tree, queryPoint, _globalOptions->k, _globalOptions->epsilon, priQueue);
            double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
            std::cout << "aprox query time: " << totalDuration << " s" << std::endl;
         }
         // naive query
         {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            std::vector<PointObjF3> knn = LinearFindKNearestNeighbors(points, queryPoint, _globalOptions->k);
            double totalDuration = duration_cast<duration<double>>(high_resolution_clock::now() - start).count();
            std::cout << "naive query time: " << totalDuration << " s" << std::endl;
         }
      }
   }
protected:
   void add_parameters(argumentum::ParameterConfig& params ) override
   {
     // ... same as above
   }
};

class TreeStatsOptions : public argumentum::CommandOptions
{
private:
   std::string inputFile;
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
private:

   template<int Dim>
   void Execute()
   {
      using namespace std::chrono;
      std::vector<PointObj<float, Dim>> points = LoadPoints<Dim>(inputFile);
      BBDTree<float, Dim> tree;
      double buildTime;
      {
         high_resolution_clock::time_point start = high_resolution_clock::now();
         tree = BBDTree<float, Dim>::BuildMidpointSplitTree(leafSize, points);
         buildTime = duration_cast<std::chrono::milliseconds>(high_resolution_clock::now() - start).count();
      }

      BBDTreeStats stats = tree.GetStats();

      int splitNodeSize = sizeof(SplitNode);
      int shrinkNodeSize = sizeof(ShrinkNode<float, Dim>);
      int leafNodeSize = sizeof(LeafNode);
      int expectedSize = splitNodeSize*stats.splitNodeCount + shrinkNodeSize*stats.shrinkNodeCount + leafNodeSize*stats.leafNodeCount;

      std::cout << "TB M NI NL NS NAP DMAX DAVG" << std::endl;

      std::cout << buildTime;
      std::cout << GetMemoryString(stats.memoryConsumption);
      std::cout << stats.innerNodeCount;
      std::cout << stats.leafNodeCount;

      std::cout << "innerNodeCount:    " << stats.innerNodeCount << std::endl;
      std::cout << "leafNodeCount:     " << stats.leafNodeCount << std::endl;
      std::cout << "splitNodeCount:    " << stats.splitNodeCount << std::endl;
      std::cout << "shrinkNodeCount:   " << stats.shrinkNodeCount << std::endl;
      std::cout << "nullCount:         " << stats.nullCount << std::endl;
      std::cout << "maxDepth:          " << stats.maxDepth << std::endl;
      std::cout << "avgDepth:          " << stats.avgDepth << std::endl;
      std::cout << "avgLeafSize:       " << stats.avgLeafSize << std::endl;
      std::cout << "memoryConsumption: " << stats.memoryConsumption << " B " << "(expected: ";
      std::cout << splitNodeSize << "*" << stats.splitNodeCount << " + ";
      std::cout << shrinkNodeSize << "*" << stats.shrinkNodeCount << " + ";
      std::cout << leafNodeSize << "*" << stats.leafNodeCount << " = ";
      std::cout << expectedSize << " B)" << std::endl;
      std::cout << "pointsConsumption: " << (sizeof(PointObj<float, Dim>) * points.size()) << " B " << std::endl;
   }

protected:
   void add_parameters(argumentum::ParameterConfig& params ) override
   {
      params.add_parameter(inputFile, "--in").nargs(1);
      params.add_parameter(dim, "--dim").nargs(1);
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
   }
};

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

class TreeVizOptions : public argumentum::CommandOptions
{
public:
   std::string inputFile;
   std::string outputFile;
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
            float shade = 0.6f * (1.0f - t) + t;

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
                  WriteSplit(file, distNode.box, half, splitDim, {1, 0, 1});
               }
               else if (node->GetType() == NodeType::SHRINK)
               {
                  const ShrinkNode<float, 3>* shrinkNode = (const ShrinkNode<float, 3>*)node;
                  leftBox = shrinkNode->GetShrinkBox();
                  WriteBox(file, leftBox, {0, 1, 0});
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
      params.add_parameter(leafSize, "--leaf").nargs(1);
      params.add_parameter(outputFile, "--out").nargs(1);
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
      params.add_parameter(dim, "--dim").nargs(1);
      params.add_parameter(k, "--k").nargs(1);
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

      for (int i_eps = 0; i_eps < 100; ++i_eps)
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

      for (int q = 0; q < 3; ++q)
      {
         FixedPriQueue<DistObj<float, Dim>>& priQueue = *fixedQueues[q];
         for (int k = 1; k <= 1024; k += std::max(1, k / 4))
         {
            double totalTime = 0;
            for (int i = 0; i < queryCount; ++i) {
               high_resolution_clock::time_point start = high_resolution_clock::now();
               std::vector<PointObj<float, Dim>> knn = FindKAproximateNearestNeighbors<float, Dim>(tree, queryPoints[i], k, epsilon, priQueue);
               double queryDuration = duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();
               totalTime += queryDuration;
            }
            double avgQueryTime = totalTime / queryCount;
            fprintf(file, "%f ", avgQueryTime);
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

   /*std::shared_ptr<GlobalOptions> globalOptions = std::make_shared<GlobalOptions>();
   std::shared_ptr<MeasureOptions> measureOptions = std::make_shared<MeasureOptions>( "measure", globalOptions );
   std::shared_ptr<StatsOptions> statsOptions = std::make_shared<StatsOptions>( "stats", globalOptions );*/
   std::shared_ptr<TreeVizOptions> treeVizOptions = std::make_shared<TreeVizOptions>("tree_viz");
   std::shared_ptr<EpsGraphOptions> epsGraphOptions = std::make_shared<EpsGraphOptions>("eps_graph");
   std::shared_ptr<QueueGraphOptions> queueGraphOptions = std::make_shared<QueueGraphOptions>("queue_graph");

   /*params.add_parameters( globalOptions );
   params.add_command(measureOptions).help("Measure times.");
   params.add_command(statsOptions).help("Measure stats.");*/
   params.add_command(treeVizOptions).help("Visualize tree hierarchy.");
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