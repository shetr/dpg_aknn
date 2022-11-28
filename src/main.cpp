#include <iostream>
#include <array>
#include <vector>
#include <stdint.h>

// udelat nejdrive uplne naivni nn a knn pro jednoduchost (linearni pruchod)
// TODO: nejdriv napsat fair split strom (FairSplitTree)
// TODO: udelat strukturu testovani a zpusob cneni vstupnich dat, zobrazovani vystupnich dat
// TODO: napsat naivni BBD strom (NaiveBBDTree)
// TODO: napsat prostorove optimalni BBD strom (TinyBBDTree)

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

// udelat kod flexibilni na zmenu fronty, dimenze a epsilon

// dotazy na Havrana:
// double nebo float?
// ruzne typy prioritnich front jsou mysleny u samotneho pruchodu pri vyhledavani nebo i u fronty kandidatu knn?
// lze na tohle pouzit to glv?

#include "aknn/vec.h"
#include "aknn/bbd_tree.h"
#include "aknn/search.h"

void BuildBBDTree()
{

}

void CreateFairSplitNode()
{

}

void CreateShrinkNode()
{

}

template<typename BBDTreeType, typename PriQueueType, typename FloatT, int Dim>
void FindAproximateNearestNeighbor(const BBDTreeType& tree, const Vec<FloatT, Dim>& point)
{

}

template<typename BBDTreeType, typename PriQueueType, typename FloatT, int Dim>
void FindKAproximateNearestNeighbors(const BBDTreeType& tree, const Vec<FloatT, Dim>& point)
{
    
}

int main()
{
    std::cout << "hi" << std::endl;

    return 0;
}