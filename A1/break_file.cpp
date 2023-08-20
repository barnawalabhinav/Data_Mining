#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

int main() {
    std::ifstream inputFile("D_large.dat");
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file." << std::endl;
        return 1;
    }

    std::vector<std::ofstream> outputFiles(5);
    for (int i = 0; i < 5; ++i) {
        outputFiles[i].open("output_" + to_string(i + 1) + ".dat");
        if (!outputFiles[i].is_open()) {
            std::cerr << "Failed to open output file " << i + 1 << "." << std::endl;
            return 1;
        }
    }

    std::string line;
    int outputFileIndex = 0;
    while (std::getline(inputFile, line)) {
        outputFiles[outputFileIndex] << line << std::endl;
        outputFileIndex = (outputFileIndex + 1) % 5;
    }

    for (int i = 0; i < 5; ++i) {
        outputFiles[i].close();
    }

    inputFile.close();

    std::cout << "File division completed." << std::endl;

    return 0;
}
