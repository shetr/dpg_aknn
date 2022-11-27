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

template<typename FloatT, int Dim>
struct Vec
{
    // values of the vector
    std::array<FloatT, Dim> v;

    const FloatT& operator[](int i) const { return v[i]; }
    FloatT& operator[](int i) { return v[i]; }
};

template<typename FloatT, int Dim>
struct AABB
{
    Vec<FloatT, Dim> min;
    Vec<FloatT, Dim> max;
};

enum class NodeType
{
    SPLIT = 0,
    SHRINK,
    LEAF
};

#define NODE_TYPE_BITS 2
#define DIM_BITS 2

#define CUSTOM_DATA_BIT_POS (DIM_BITS + NODE_TYPE_BITS)
#define NODE_TYPE_MASK ((1 << NODE_TYPE_BITS) - 1)
#define DIM_MASK ((1 << DIM_BITS) - 1)

struct Node
{
    uint64_t customData_nodeType;
};

// data up to 10^8
// build from left
// size 8 bytes
struct SplitNode
{
    // 60 bits right child index, 2 bits dimension, 2 bits node type
    uint64_t rightChild_splitDim_nodeType;
};

// size 8 + 2*sizeof(Vec<FloatT, Dim>) bytes
// float:  8 +  8*Dim
// double: 8 + 16*Dim
template<typename FloatT, int Dim>
struct ShrinkNode
{
    uint64_t rightChild_nodeType;
    AABB<FloatT, Dim> shrinkBox;
};

// size 16 bytes
template<typename FloatT, int Dim>
struct LeafNode
{
    uint64_t pointsFrom_nodeType;
    uint64_t pointsTo;
};

struct KDTree
{

};

template<typename FloatT, int Dim>
struct BBDTree
{
    std::vector<Node> nodes;
    std::vector<Vec<FloatT, Dim>> points;
    AABB pointsBox;

    Node* GetRoot();
};

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

    return 0;
}