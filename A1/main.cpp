#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <queue>
#include <string>
#include <bits/stdc++.h>

using namespace std;
class TreeNode
{
public:
    int item = -1;
    int count;
    int copy_count;
    TreeNode *parent;
    map<int, TreeNode *> children;

    TreeNode(int _item, int _count, TreeNode *_parent = nullptr)
        : item(_item), count(_count), parent(_parent) {}
};

map<string, int> *encoding = new map<string, int>();
int value = -1;

TreeNode *buildTree(const vector<vector<int>> &transactions)
{
    unordered_map<int, int> *frequency = new unordered_map<int, int>();
    for (const auto &transaction : transactions)
        for (int item : transaction)
            (*frequency)[item]++;

    vector<int> *frequentItems = new vector<int>();
    for (const auto &entry : (*frequency))
        (*frequentItems).push_back(entry.first);

    sort((*frequentItems).begin(), (*frequentItems).end(), [&](int a, int b)
         { return (*frequency)[a] > (*frequency)[b]; });

    TreeNode *root = new TreeNode(-1, 0);
    for (const auto &transaction : transactions)
    {
        vector<int> *filteredTransaction = new vector<int>();
        for (int item : transaction)
            if (find((*frequentItems).begin(), (*frequentItems).end(), item) != (*frequentItems).end())
                (*filteredTransaction).push_back(item);

        sort((*filteredTransaction).begin(), (*filteredTransaction).end(), [&](int a, int b)
             { return (*frequency)[a] > (*frequency)[b] || ((*frequency)[a] == (*frequency)[b] && a < b); });

        TreeNode *node = root;
        for (int item : (*filteredTransaction))
        {
            if (node->children.find(item) != node->children.end())
            {
                node = node->children[item];
                node->count++;
                node->copy_count++;
            }
            else
            {
                TreeNode *child = new TreeNode(item, 1, node);
                node->children[item] = child;
                node = child;
            }
        }
    }
    delete frequency;
    delete frequentItems;
    return root;
}

bool encodeTree(TreeNode *node, int min_support, string prefix)
{
    if (node->children.size() == 0)
        return false;

    if (node->item >= 0)
        prefix += to_string(node->item) + " ";

    for (const auto &childPair : node->children)
    {
        TreeNode *child = childPair.second;
        if (node->copy_count < min_support && node->item >= 0)
            break;
        if (encodeTree(child, min_support, prefix))
            node->copy_count -= child->count;
    }
    if (node->copy_count >= min_support && node->item >= 0 && count(prefix.begin(), prefix.end(), ' ') > 1)
    {
        (*encoding)[prefix] = value--;
        return true;
    }
    return false;
}

void write_mapping(ofstream &outFile)
{
    outFile << (*encoding).size() << endl;
    for (auto e : (*encoding))
        outFile << e.second << "\n"
                << e.first << endl;
}

void processTransaction(string prefix, int freq, ofstream &outFile)
{
    string ans = "";
    int l = 0, r = prefix.size() - 1;
    while (l < r)
    {
        string temp = prefix.substr(l, r - l + 1);
        if ((*encoding).find(temp) != (*encoding).end())
        {
            if (ans == "")
                ans = to_string((*encoding)[temp]);
            else
                ans += " " + to_string((*encoding)[temp]);
            l = r + 1;
            r = prefix.size() - 1;
            continue;
        }
        r--;
        while (r > l && prefix[r] != ' ')
            r--;
    }
    if (l < prefix.size() - 1)
        if (ans == "")
            ans = prefix.substr(l, prefix.size() - l - 1);
        else
            ans += " " + prefix.substr(l, prefix.size() - l - 1);

    if (outFile.is_open())
        while (freq--)
            outFile << ans << "\n";
    else
        std::cerr << "Failed to open the output file." << std::endl;
}

void mineTree(TreeNode *node, int min_support, string prefix, ofstream &outfile)
{
    if (node->item >= 0)
        prefix += to_string(node->item) + " ";

    for (const auto &childPair : node->children)
    {
        TreeNode *child = childPair.second;
        node->count -= child->count;
        mineTree(child, min_support, prefix, outfile);
    }

    if (node->count > 0)
        processTransaction(prefix, node->count, outfile);
}

void printTree(TreeNode *node, int level)
{
    if (node->children.size() == 0)
        return;

    queue<TreeNode *> q;
    q.push(node);
    std::cout << node->item << " (" << node->count << ", " << node->copy_count << ")" << endl;
    while (!q.empty())
    {
        TreeNode *curnode = q.front();
        q.pop();
        for (auto childnode : curnode->children)
        {
            std::cout << childnode.first << " (" << childnode.second->count << ", " << childnode.second->copy_count << ")"
                      << " && ";
            q.push(childnode.second);
        }
        std::cout << endl;
    }

    std::cout << "------------------------------------\n";
    for (auto &elem : (*encoding))
        std::cout << elem.first << " " << elem.second << endl;
}

int decompress(string compressedPath, string outputPath)
{
    unordered_map<int, string> *fileEncoding = new unordered_map<int, string>();
    ifstream compressedFile(compressedPath);
    if (!compressedFile.is_open())
    {
        cerr << "Failed to open the file." << endl;
        return 1;
    }

    ofstream outFile = ofstream(outputPath);
    string line;
    int lineNum = 0, numEncodings = 0, lastEncoding = 0;
    while (getline(compressedFile, line))
    {
        if (lineNum == 0)
            numEncodings = stoi(line);
        else if (lineNum <= 2 * numEncodings)
            if (lineNum & 1)
                lastEncoding = stoi(line);
            else
                (*fileEncoding)[lastEncoding] = line;
        else
        {
            istringstream tokenizer(line);
            string token;

            while (tokenizer >> token)
            {
                int element = stoi(token);
                if (element < 0)
                    outFile << (*fileEncoding)[element];
                else
                    outFile << element << " ";
            }
            outFile << "\n";
        }
        lineNum++;
    }
    outFile.close();
    delete fileEncoding;
    return 0;
}

int compress(string dataPath, string outputPath)
{
    vector<vector<int>> *transactions = new vector<vector<int>>();

    ifstream inputFile(dataPath);
    if (!inputFile.is_open())
    {
        cerr << "Failed to open the file." << endl;
        return 1;
    }
    string line;
    while (getline(inputFile, line))
    {
        vector<int> tokens;
        istringstream tokenizer(line);
        string token;

        while (tokenizer >> token)
            tokens.push_back(stoi(token));

        transactions->push_back(tokens);
    }
    int minSupport = 2;
    TreeNode *root = buildTree(*transactions);
    bool flag = encodeTree(root, minSupport, "");

    ofstream outFile(outputPath);
    write_mapping(outFile);
    mineTree(root, minSupport, "", outFile);
    outFile.close();

    // Remember to free the allocated memory to avoid memory leaks
    // You can create a function to delete the tree nodes recursively
    delete transactions;
    delete encoding;
    delete root;

    return 0;
}

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

float compressionRatio(string compressedFile, string originalFile)
{
    ifstream cf(compressedFile);
    ifstream of(originalFile);
    string line;
    int sizeCf = 0, sizeOf = 0;

    while (getline(cf, line))
    {
        istringstream tokenizer(line);
        string token;
        while (tokenizer >> token)
            sizeCf++;
    }
    while (getline(of, line))
    {
        istringstream tokenizer(line);
        string token;
        while (tokenizer >> token)
            sizeOf++;
    }

    cout << "sizeCf = " << sizeCf << endl;
    cout << "sizeOf = " << sizeOf << endl;
    return (float)sizeCf / (float)sizeOf;
}

int main(int argc, char *argv[])
{
    char *dataPath;
    char *outputPath;
    dataPath = argv[2];
    outputPath = argv[3];

    if (strcmp(argv[1], "C") == 0)
    {
        compress(dataPath, outputPath);
        cout << "Compression Ratio: " << compressionRatio(outputPath, dataPath) << "\n";
    }
    // return compress(dataPath, outputPath);
    else
    {
        decompress(dataPath, outputPath);
        cout << "Loss: " << isLossLess(dataPath, outputPath) << "\n";
    }
    // return decompress(dataPath, outputPath);
}