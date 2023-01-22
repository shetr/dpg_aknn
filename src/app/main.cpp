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

class GlobalOptions : public argumentum::Options
{
public:
   int logLevel = 0;
   void add_parameters( ParameterConfig& params ) override
   {
      params.add_parameter( logLevel, "--loglevel" ).nargs( 1 );
   }
};

class AccumulatorOptions : public argumentum::CommandOptions
{
   std::shared_ptr<GloblaOptions> mpGlobal;
public:
   AccumulatorOptions( std::string_view name, std::shared_ptr<GloblaOptions> pGlobal )
      : CommandOptions( name )
      , mpGlobal( pGlobal )
  {}

  void execute( const ParseResults& res )
  {
     if ( mpGlobal && mpGlobal->logLevel > 0 )
       cout << "Accumulating " << numbers.size() << " numbers\n";

     auto acc = accumulate(
        numbers.begin(), numbers.end(), operation.second, operation.first );
     cout << acc << "\n";
  }
};

int main(int argc, char** argv)
{
    std::cout << "hi" << std::endl;

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
    
    //std::cout << (std::numeric_limits<float>::infinity() / 2) << std::endl;

    return 0;
}