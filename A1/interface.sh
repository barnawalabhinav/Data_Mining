if [ "$1" == "C" ]; then
    ./main.o C "$2" "$3"
else
    ./main.o D "$2" "$3"
fi