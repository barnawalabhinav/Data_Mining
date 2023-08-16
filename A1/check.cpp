
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

bool isLossLess(string file1, string file2)
{
    ifstream f1(file1);
    ifstream f2(file2);
    string line1, line2;
    vector<vector<int>> *v1 = new vector<vector<int>>();
    vector<vector<int>> *v2 = new vector<vector<int>>();
    while (getline(f1, line1) && getline(f2, line2))
    {
        vector<int> trans1, trans2;
        istringstream tokenizer1(line1);
        string token;
        while (tokenizer1 >> token)
            trans1.push_back(stoi(token));

        istringstream tokenizer2(line2);
        while (tokenizer2 >> token)
            trans2.push_back(stoi(token));
        sort(trans1.begin(), trans1.end());
        sort(trans2.begin(), trans2.end());
        v1->push_back(trans1);
        v2->push_back(trans2);
    }
    if (getline(f1, line1) || getline(f2, line2))
        return false;

    sort(v1->begin(), v1->end());
    sort(v2->begin(), v2->end());
    for (int i = 0; i < v1->size(); i++)
        for (int j = 0; j < (*v1)[i].size(); j++)
            if ((*v1)[i][j] != (*v2)[i][j])
                return false;
    return true;
}

int main(int argc, char *argv[])
{
    char *file1Path;
    char *file2Path;
    file1Path = argv[1];
    file2Path = argv[2];

    if (isLossLess(file1Path, file2Path))
        cout << "NO Loss" << endl;
    else
        cout << "Lossy" << endl;

    // bash compile.sh && bash interface.sh D out.txt final_out.txt && check.o A1_datasets/test.dat final_out.txt
}