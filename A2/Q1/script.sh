#!/bin/bash

python format.py $1 > temp.txt_graph

chmod 777 gaston-1.1/gaston
{ time gaston-1.1/gaston 60905 temp.txt_graph -p ; } 2> time_gaston_95.txt
{ time gaston-1.1/gaston 32055 temp.txt_graph -p ; } 2> time_gaston_50.txt
{ time gaston-1.1/gaston 16028 temp.txt_graph -p ; } 2> time_gaston_25.txt
{ time gaston-1.1/gaston 6411 temp.txt_graph -p ; } 2> time_gaston_10.txt
{ time gaston-1.1/gaston 3206 temp.txt_graph -p ; } 2> time_gaston_05.txt

chmod 777 gSpan6/gSpan-64
{ time gSpan6/gSpan-64 -f temp.txt_graph -s 0.95 ; } 2> time_gspan_95.txt
{ time gSpan6/gSpan-64 -f temp.txt_graph -s 0.50 ; } 2> time_gspan_50.txt
{ time gSpan6/gSpan-64 -f temp.txt_graph -s 0.25 ; } 2> time_gspan_25.txt
{ time gSpan6/gSpan-64 -f temp.txt_graph -s 0.10 ; } 2> time_gspan_10.txt
{ time gSpan6/gSpan-64 -f temp.txt_graph -s 0.05 ; } 2> time_gspan_05.txt

chmod 777 pafi-1.0.1/Linux/fsg
{ time pafi-1.0.1/Linux/fsg -s 95 temp.txt_graph ; } 2> time_pafi_95.txt
{ time pafi-1.0.1/Linux/fsg -s 50 temp.txt_graph ; } 2> time_pafi_50.txt
{ time pafi-1.0.1/Linux/fsg -s 25 temp.txt_graph ; } 2> time_pafi_25.txt
{ time pafi-1.0.1/Linux/fsg -s 10 temp.txt_graph ; } 2> time_pafi_10.txt
{ time pafi-1.0.1/Linux/fsg -s 5 temp.txt_graph ; } 2> time_pafi_05.txt

python graph.py

rm temp.txt_graph
rm *.fp
rm time_gspan_*.txt
rm time_gaston_*.txt
rm time_pafi_*.txt