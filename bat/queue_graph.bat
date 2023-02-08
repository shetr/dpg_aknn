exe\app.exe queue_graph --in data\clusters_3d_e5.txt --out out\queue_graph.txt --dim 3 --eps 0 --leaf 10
python src\scripts\queue_graph.py out\queue_graph.txt