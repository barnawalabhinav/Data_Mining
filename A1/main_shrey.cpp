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
#include <chrono>

using namespace std;
class TreeNode
{
public:
    int item = -1;
    int count;
    int copy_count=0;
    int cc_count=0;
    TreeNode *parent;
    map<int, TreeNode *> children;

    TreeNode(int _item, int _count, TreeNode *_parent = nullptr)
        : item(_item), count(_count), cc_count(_count), copy_count(_count), parent(_parent) {}
};

map<string, int> *encoding = new map<string, int>();
vector<int> *encoding_used = new vector<int>();
int value = -1, num_transactions = 0;

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
                node->cc_count++;
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

TreeNode *buildTree_copy(const vector<vector<int>> &transactions)
{
    unordered_map<int, int> *frequency = new unordered_map<int, int>();
    for (const auto &transaction : transactions)
        for (int item : transaction)
            (*frequency)[item]++;

    vector<int> *frequentItems = new vector<int>();
    for (const auto &entry : (*frequency))
        (*frequentItems).push_back(entry.first);

    sort((*frequentItems).begin(), (*frequentItems).end(), [&](int a, int b)
         { return (*frequency)[a] < (*frequency)[b]; });

    TreeNode *root = new TreeNode(-1, 0);
    for (const auto &transaction : transactions)
    {
        vector<int> *filteredTransaction = new vector<int>();
        for (int item : transaction)
            if (find((*frequentItems).begin(), (*frequentItems).end(), item) != (*frequentItems).end())
                (*filteredTransaction).push_back(item);

        sort((*filteredTransaction).begin(), (*filteredTransaction).end(), [&](int a, int b)
             { return (*frequency)[a] < (*frequency)[b] || ((*frequency)[a] == (*frequency)[b] && a > b); });

        TreeNode *node = root;
        for (int item : (*filteredTransaction))
        {
            if (node->children.find(item) != node->children.end())
            {
                node = node->children[item];
                node->count++;
                node->copy_count++;
                node->cc_count++;
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

int encodeTree(TreeNode *node, int min_support, string prefix)
{
    if (node->children.size() == 0)
        return -1;

    if (node->item >= 0)
        prefix += to_string(node->item) + " ";
    
    int node_copy_count = node->copy_count;

    for (const auto &childPair : node->children)
    {
        TreeNode *child = childPair.second;
        if (node_copy_count < min_support && node->item >= 0)
            break;
        int child_count = encodeTree(child, min_support, prefix);
        if (child_count!=-1)
            node_copy_count -= child_count;
    }
    if (node_copy_count >= min_support && node->item >= 0 && count(prefix.begin(), prefix.end(), ' ') > 1)
    {
        (*encoding)[prefix] = value--;
        return node_copy_count;
    }
    return -1;
}

int encodeTree_copy(TreeNode *node, int min_support, string prefix)
{
    if (node->children.size() == 0)
        return -1;

    if (node->item >= 0)
        prefix = to_string(node->item) + " " + prefix;
    
    int node_copy_count = node->copy_count;

    for (const auto &childPair : node->children)
    {
        TreeNode *child = childPair.second;
        if (node_copy_count < min_support && node->item >= 0)
            break;
        int child_count = encodeTree_copy(child, min_support, prefix);
        if (child_count!=-1)
            node_copy_count -= child_count;
    }
    if (node_copy_count >= min_support && (*encoding).find(prefix) == (*encoding).end() && node->item >= 0 && count(prefix.begin(), prefix.end(), ' ') > 1)
    {
        // cout<<prefix<<" : "<<value<<endl;
        (*encoding)[prefix] = value--;
        return node_copy_count;
    }
    return -1;
}

void write_mapping(ofstream &outFile)
{
    int size = 0;
    for (auto e : (*encoding_used))
        if (e)
            size++;
    outFile << "0 " << size << endl;

    for (auto e : (*encoding))
        if ((*encoding_used)[-e.second]>1){
            if(count(e.first.begin(), e.first.end(), ' ') == 1 and (*encoding_used)[-e.second] == 2)
                continue;
            outFile << e.second << "\n"
                    << e.first << endl;
        }
}

void processTransaction_1(string prefix, int freq, ofstream &outFile)
{
        /************ DP based Encoding Starts here ************/

    int l = 0, r = prefix.size() - 1;
    while(l<prefix.size()-1)
    {    
        if(prefix[l]==' ') l++;
        while (l < r)
        {
            string temp = prefix.substr(l, r - l + 1);
            if ((*encoding).find(temp) != (*encoding).end())
            {
                (*encoding_used)[-(*encoding)[temp]] += freq;

                l = r + 1;
                r = prefix.size() - 1;
                continue;
            }
            r--;
            while (r > l && prefix[r] != ' ')
                r--;
        }
        int end = l+1; while(end<prefix.size() and prefix[end]!=' ') end++;
        l=end+1; r = prefix.size() - 1;
    }

// ************************ Greedy normal

    // int l = 0, r = prefix.size() - 1;
    // while (l < r)
    // {
    //     string temp = prefix.substr(l, r - l + 1);
    //     if ((*encoding).find(temp) != (*encoding).end())
    //     {
    //         (*encoding_used)[-(*encoding)[temp]] += freq;
    //         l = r + 1;
    //         r = prefix.size() - 1;
    //         continue;
    //     }
    //     r--;
    //     while (r > l && prefix[r] != ' ')
    //         r--;
    // }
}

void processTransaction_2(string prefix, int freq, ofstream &outFile)
{
    string ans = "";
    if(freq > 1) 
        ans="0"+to_string(freq);

        /************ DP based Encoding Starts here ************/

    int l = 0, r = prefix.size() - 1;
    while(l<prefix.size()-1)
    {    
        if(prefix[l]==' ') l++;
        while (l < r)
        {
            string temp = prefix.substr(l, r - l + 1);
            if ((*encoding).find(temp) != (*encoding).end() and (*encoding_used)[-(*encoding)[temp]] > 1)
            {
                if(count(temp.begin(), temp.end(), ' ') == 1 and (*encoding_used)[-(*encoding)[temp]] == 2)
                   continue;
                ans += " " + to_string((*encoding)[temp]);
                l = r + 1;
                r = prefix.size() - 1;
                continue;
            }
            r--;
            while (r > l && prefix[r] != ' ')
                r--;
        }
        int end = l+1; while(end<prefix.size() and prefix[end]!=' ') end++;
        ans += " " + prefix.substr(l, end-l);
        l=end+1; r = prefix.size() - 1;
    }
    if(ans[0]==' ') ans=ans.substr(1);

// ************************ Greedy normal

    // int l = 0, r = prefix.size() - 1;
    // while (l < r)
    // {
    //     string temp = prefix.substr(l, r - l + 1);
    //     if ((*encoding).find(temp) != (*encoding).end() and (*encoding_used)[-(*encoding)[temp]] > 1)
    //     {
    //         if(count(temp.begin(), temp.end(), ' ') == 1 and (*encoding_used)[-(*encoding)[temp]] == 2)
    //             continue;
    //         if (ans == "")
    //             ans = to_string((*encoding)[temp]);
    //         else
    //             ans += " " + to_string((*encoding)[temp]);
    //         l = r + 1;
    //         r = prefix.size() - 1;
    //         continue;
    //     }
    //     r--;
    //     while (r > l && prefix[r] != ' ')
    //         r--;
    // }
    // if (l < prefix.size() - 1)
    //     if (ans == "")
    //         ans = prefix.substr(l, prefix.size() - l - 1);
    //     else
    //         ans += " " + prefix.substr(l, prefix.size() - l - 1);

    // if(ans[0]==' ') ans=ans.substr(1);

    if (outFile.is_open())
        // while (freq--)
            outFile << ans << "\n";
    else
        std::cerr << "Failed to open the output file." << std::endl;
}

void mineTree_1(TreeNode *node, int min_support, string prefix, ofstream &outfile)
{
    if(node == nullptr) return;
    if (node->item >= 0)
        prefix += to_string(node->item) + " ";

    for (const auto &childPair : node->children)
    {
        TreeNode *child = childPair.second;
        node->count -= child->count;
        mineTree_1(child, min_support, prefix, outfile);
    }

// cout<<prefix<<" ^ "<<node->count<<endl;

    if (node->count > 0)
        processTransaction_1(prefix, node->count, outfile);
}

void mineTree_2(TreeNode *node, int min_support, string prefix, ofstream &outfile)
{
    if(node == nullptr) return;
    if (node->item >= 0)
        prefix += to_string(node->item) + " ";

    for (const auto &childPair : node->children)
    {
        TreeNode *child = childPair.second;
        node->cc_count -= child->cc_count;
        mineTree_2(child, min_support, prefix, outfile);
    }

    // cout<<prefix<<" % "<<node->cc_count<<endl;

    if (node->cc_count > 0)
        processTransaction_2(prefix, node->cc_count, outfile);
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
    vector<string> *transactions = new vector<string>();
    ifstream compressedFile(compressedPath);
    if (!compressedFile.is_open())
    {
        cerr << "Failed to open the file." << endl;
        return 1;
    }

    ofstream outFile = ofstream(outputPath);
    string line;
    bool flag = true;
    int lineNum = 0, numEncodings = 0, lastEncoding = 0;
    while (getline(compressedFile, line))
    {
        if (flag)
        {
            istringstream tokenizer(line);
            string token;
            tokenizer >> token;
            if(token=="") continue;
            if (stoi(token) == 0)
            {
                flag = false;
                tokenizer >> token;
                numEncodings = stoi(token);
            }
            else if(token[0]=='0'){
                int x = 0;
                while(line[x]!=' ') x++;
                x++;
                for(int i=stoi(token); i; i--)
                    (*transactions).push_back(line.substr(x));
            }
            else 
                (*transactions).push_back(line);
        }
        else
        {
            lineNum++;
            if (lineNum & 1)
                lastEncoding = stoi(line);
            else
                (*fileEncoding)[lastEncoding] = line;
        }
    }
    compressedFile.close();

    for (auto e : (*transactions))
    {
        istringstream tokenizer(e);
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
    outFile.close();
    delete fileEncoding;
    return 0;
}

void deleteNodes(TreeNode *node)
{
    for (auto child : node->children)
        deleteNodes(child.second);
    delete node;
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
        num_transactions++;
        vector<int> tokens;
        istringstream tokenizer(line);
        string token;

        while (tokenizer >> token)
            tokens.push_back(stoi(token));

        transactions->push_back(tokens);
    }
    int minSupport = 10;
    cout << "Min Support: " << minSupport<<endl;
    std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    TreeNode *root = buildTree(*transactions);
    std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    std::cout << "Time for Build Tree 1: " << elapsedTime.count() << " seconds" << std::endl;
    startTime = std::chrono::system_clock::now();
    TreeNode *root_copy = buildTree_copy(*transactions);
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Build Tree 2: " << elapsedTime.count() << " seconds" << std::endl;

    delete transactions;
    startTime = std::chrono::system_clock::now();
    int flag = encodeTree(root, minSupport, "");
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Encode Tree 1: " << elapsedTime.count() << " seconds" << std::endl;

    cout<<"Map size before: "<<(*encoding).size()<<endl;

    startTime = std::chrono::system_clock::now();
    int flag_copy = encodeTree_copy(root_copy, minSupport, "");
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Encode Tree 2: " << elapsedTime.count() << " seconds" << std::endl;

    cout<<"Map size after: "<<(*encoding).size()<<endl;

    (*encoding_used).resize(-value + 1);

    ofstream outFile(outputPath);
    startTime = std::chrono::system_clock::now();
    mineTree_1(root, minSupport, "", outFile);
    cout<<"mine 1 done\n";
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Mine Tree 1: " << elapsedTime.count() << " seconds" << std::endl;
    startTime = std::chrono::system_clock::now();
    mineTree_2(root, max(5, minSupport/4), "", outFile);
    cout<<"mine 2 done\n";
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Mine Tree 2: " << elapsedTime.count() << " seconds" << std::endl;
    write_mapping(outFile);
    outFile.close();

    cout<<"Compress Done\n";

    // Remember to free the allocated memory to avoid memory leaks
    // You can create a function to delete the tree nodes recursively
    delete encoding;
    // deleteNodes(root);

    return 0;
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

    // cout << "sizeCf = " << sizeCf << endl;
    // cout << "sizeOf = " << sizeOf << endl;
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
        cout << "Compression Ratio:\n" << compressionRatio(outputPath, dataPath) << "\n";
    }
    // return compress(dataPath, outputPath);
    else
        return decompress(dataPath, outputPath);
}