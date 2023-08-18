bash compile.sh
g++ check.cpp -o check.o
bash interface.sh C "$1" out.txt
bash interface.sh D out.txt final_out.txt
./check.o "$1" final_out.txt
# rm final_out.txt
# rm out.txt
# rm check.o
# rm main.o