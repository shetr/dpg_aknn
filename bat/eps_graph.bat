exe\app.exe eps_graph --in data\clusters_3d_e5.txt --out out\eps_graph.txt --dim 3 --k 1 --leaf 10 --queries 10000
python src\scripts\eps_graph.py out\eps_graph.txt