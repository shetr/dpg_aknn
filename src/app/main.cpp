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
      params.add_parameter( inputFile, "--in" ).nargs( 1 );
      params.add_parameter( dim, "--dim" ).nargs( 1 );
      params.add_parameter( k, "--k" ).nargs( 1 );
      params.add_parameter( epsilon, "--eps" ).nargs( 1 );
      params.add_parameter( leafSize, "--leaf" ).nargs( 1 );
      params.add_parameter( outputFile, "--out" ).nargs( 1 );
   }
};

std::vector<PointObj<float, 3>> LoadPoints(const std::string& filename)
{
   std::vector<PointObj<float, 3>> res;
   FILE* file = fopen(filename.c_str(), "r");
   if (!file) {
      std::cout << "failed to read file: " << filename << std::endl;
      return res;
   }
   Vec<float, 3> v;
   while (fscanf(file, "%f %f %f\n", &v[0], &v[1], &v[2]) != EOF)
   {
      res.push_back(PointObj<float, 3>({v}));
   }
   
   return res;
}

void SavePoints(const std::string& filename, const std::vector<PointObj<float, 3>>& points)
{

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
         std::vector<PointObj<float, 3>> points = LoadPoints(_globalOptions->inputFile);
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
};

class StatsOptions : public argumentum::CommandOptions
{
   std::shared_ptr<GlobalOptions> _globalOptions;
public:
   StatsOptions( std::string_view name, std::shared_ptr<GlobalOptions> globalOptions)
      : CommandOptions(name), _globalOptions(globalOptions)
   {}

   void execute(const argumentum::ParseResult& res)
   {
      using namespace std::chrono;
      if (_globalOptions && _globalOptions->inputFile.size() > 0)
      {
         std::vector<PointObj<float, 3>> points = LoadPoints(_globalOptions->inputFile);
         BBDTree<float, 3> tree;
         //tree = BBDTree<float, 3>::BuildBasicSplitTree(_globalOptions->leafSize, points);
         tree = BBDTree<float, 3>::BuildMidpointSplitTree(_globalOptions->leafSize, points);
         BBDTreeStats stats = tree.GetStats();

         int splitNodeSize = sizeof(SplitNode);
         int shrinkNodeSize = sizeof(ShrinkNode<float, 3>);
         int leafNodeSize = sizeof(LeafNode);
         int expectedSize = splitNodeSize*stats.splitNodeCount + shrinkNodeSize*stats.shrinkNodeCount + leafNodeSize*stats.leafNodeCount;

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
         std::cout << "pointsConsumption: " << (sizeof(PointObj<float, 3>) * points.size()) << " B " << std::endl;
      }
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

class VisualizeOptions : public argumentum::CommandOptions
{
   std::shared_ptr<GlobalOptions> _globalOptions;
public:
   VisualizeOptions( std::string_view name, std::shared_ptr<GlobalOptions> globalOptions)
      : CommandOptions(name), _globalOptions(globalOptions)
   {}

   void execute(const argumentum::ParseResult& res)
   {
      using namespace std::chrono;
      if (_globalOptions && _globalOptions->inputFile.size() > 0 && _globalOptions->outputFile.size() > 0)
      {
         std::vector<PointObj<float, 3>> points = LoadPoints(_globalOptions->inputFile);
         BBDTree<float, 3> tree;
         //tree = BBDTree<float, 3>::BuildBasicSplitTree(_globalOptions->leafSize, points);
         tree = BBDTree<float, 3>::BuildMidpointSplitTree(_globalOptions->leafSize, points);

         BBDTreeStats stats = tree.GetStats();

         FILE* file = fopen(_globalOptions->outputFile.c_str(), "w");
         if (!file) {
            std::cout << "failed to write to a file " << _globalOptions->outputFile << std::endl;
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
            float shade = 0.2f * (1.0f - t) + t;

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
};

int main(int argc, char** argv)
{
   using namespace argumentum;

   argument_parser parser;
   ParameterConfig params = parser.params();
   parser.config().program( argv[0] ).description( "Aproximate k nearest neighbor search" );

   std::shared_ptr<GlobalOptions> globalOptions = std::make_shared<GlobalOptions>();
   std::shared_ptr<MeasureOptions> measureOptions = std::make_shared<MeasureOptions>( "measure", globalOptions );
   std::shared_ptr<StatsOptions> statsOptions = std::make_shared<StatsOptions>( "stats", globalOptions );
   std::shared_ptr<VisualizeOptions> visOptions = std::make_shared<VisualizeOptions>( "viz_tree", globalOptions );

   params.add_parameters( globalOptions );
   params.add_command(measureOptions).help("Measure times.");
   params.add_command(statsOptions).help("Measure stats.");
   params.add_command(visOptions).help("Visualize tree hierarchy.");

   ParseResult res = parser.parse_args( argc, argv, 1 );
   if ( !res )
      return 1;

   auto pcmd = res.commands.back();
   if ( !pcmd )
      return 1;

   pcmd->execute( res );
/*
   std::vector<PointObj<float, 2>> points;
   points.push_back({Vec<float, 2>({0, 0})});
   points.push_back({Vec<float, 2>({1, 0})});
   points.push_back({Vec<float, 2>({0, 1})});
   points.push_back({Vec<float, 2>({1, 1})});

   Vec<float, 2> query({0.f, 0.75f});
   PointObj<float, 2> nn = LinearFindNearestNeighbor(points, query);

   std::cout << nn.point[0] << ", " << nn.point[1] << std::endl;
   
   std::vector<PointObj<float, 2>> knn = LinearFindKNearestNeighbors(points, query, 3);

   for (int i = 0; i < (int)knn.size(); ++i) {
      std::cout << knn[i].point[0] << ", " << knn[i].point[1] << std::endl;
   }
   
   std::cout << (std::numeric_limits<float>::infinity() / 2) << std::endl;
*/
   return 0;
}