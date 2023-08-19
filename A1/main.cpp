#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <queue>
#include <string>
#include <chrono>

using namespace std;
class TreeNode
{
public:
    int item = -1;
    long long count = 0;
    long long copy_count = 0;
    map<int, TreeNode *> children;

    TreeNode(int _item = -1, long long _count = 0, map<int, TreeNode *> _children = map<int, TreeNode *>())
        : item(_item), count(_count), copy_count(_count) {}
};

map<string, int> *encoding = new map<string, int>();
vector<bool> *encoding_used = new vector<bool>();
int value = -1, num_transactions = 0;

TreeNode *buildTree(const vector<vector<int>> *transactions)
{
    unordered_map<int, long long> *frequency = new unordered_map<int, long long>();
    for (const auto &transaction : *transactions)
        for (int item : transaction)
            (*frequency)[item]++;

    vector<int> *frequentItems = new vector<int>();
    for (const auto &entry : (*frequency))
        (*frequentItems).push_back(entry.first);

    sort((*frequentItems).begin(), (*frequentItems).end(), [&](int a, int b)
         { return (*frequency)[a] > (*frequency)[b]; });

    TreeNode *root = new TreeNode(-1, 0);
    vector<int> *filteredTransaction;
    for (const auto &transaction : *transactions)
    {
        filteredTransaction = new vector<int>();
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
                TreeNode *child = new TreeNode(item, 1);
                node->children[item] = child;
                node = child;
            }
        }
    }
    delete filteredTransaction;
    delete frequency;
    delete frequentItems;
    return root;
}

void mergeTree(TreeNode *from_node, TreeNode *to_node)
{
    // cout << "Merging " << from_node->item << " to " << to_node->item << "\n";
    // TreeNode *to_node = to_root->children[from_node->item];
    to_node->count += from_node->count;
    to_node->copy_count += from_node->copy_count;
    for (const auto childPair : from_node->children)
    {
        TreeNode *child = childPair.second;
        if (to_node->children.find(child->item) != to_node->children.end())
            mergeTree(child, to_node->children[child->item]);
        else
        {
            TreeNode *newChild = new TreeNode(child->item, child->count, child->children);
            to_node->children[newChild->item] = newChild;
        }
    }
}

bool encodeTree(TreeNode *node, int min_support, string prefix, TreeNode *newTreeRoot, bool buildResidual)
{
    // cout << node->item << " " << node->copy_count << " " << node->children.size() << "\n";

    if (node->children.size() == 0)
        return false;

    if (node->item >= 0)
        prefix += to_string(node->item) + " ";

    for (const auto childPair : node->children)
    {
        TreeNode *child = childPair.second;
        if (node->copy_count < min_support && node->item >= 0)
        {
            // cout << node->copy_count << "\n";
            if (!buildResidual)
                break;

            if (newTreeRoot->children.find(child->item) != newTreeRoot->children.end())
                mergeTree(child, newTreeRoot->children[child->item]);
            else
            {
                TreeNode *newChild = new TreeNode(child->item, child->count, child->children);
                newTreeRoot->children[newChild->item] = newChild;
            }
            continue;
        }
        if (encodeTree(child, min_support, prefix, newTreeRoot, buildResidual))
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
    int size = 0;
    for (auto e : (*encoding_used))
        if (e)
            size++;
    outFile << "0 " << size << endl;

    for (auto e : (*encoding))
        if ((*encoding_used)[-e.second])
            outFile << e.second << "\n"
                    << e.first << endl;
}

void processTransaction(string prefix, long long freq, ofstream &outFile)
{
    string ans = "";

    /************ DP based Encoding Starts here ************/

    // int size = count(prefix.begin(), prefix.end(), ' ');
    // int dp[size + 1];
    // dp[size] = 0;
    // for (int i = size - 1; i >= 0; i--)
    // {
    //     dp[i] = size - i;
    //     string code = "";
    //     int cnt = 0, pos = 0;
    //     while (cnt < i)
    //     {
    //         if (prefix[pos] == ' ')
    //             cnt++;
    //         pos++;
    //     }
    //     int ptr = pos;
    //     for (int j = i; j < size; j++)
    //     {
    //         while (ptr < prefix.size() && prefix[ptr] != ' ')
    //         {
    //             code.push_back(prefix[ptr]);
    //             ptr++;
    //         }
    //         code.push_back(' ');
    //         ptr++;
    //         if ((*encoding).find(code) != (*encoding).end())
    //             dp[i] = min(dp[i], dp[j + 1] + 1);
    //         else
    //             dp[i] = min(dp[i], dp[j + 1] + j - i + 1);
    //     }
    // }

    // int i = 0;
    // while (i < size)
    // {
    //     string code = "";
    //     int cnt = 0, pos = 0;
    //     while (cnt < i)
    //     {
    //         if (prefix[pos] == ' ')
    //             cnt++;
    //         pos++;
    //     }
    //     int ptr = pos, j = i;
    //     for (; j < size; j++)
    //     {
    //         while (ptr < prefix.size() && prefix[ptr] != ' ')
    //         {
    //             code.push_back(prefix[ptr]);
    //             ptr++;
    //         }
    //         code.push_back(' ');
    //         ptr++;
    //         if ((*encoding).find(code) != (*encoding).end() && dp[i] == dp[j + 1] + 1)
    //         {
    //             (*encoding_used)[-(*encoding)[code]] = true;
    //             if (ans == "")
    //                 ans = to_string((*encoding)[code]);
    //             else
    //                 ans += " " + to_string((*encoding)[code]);
    //             i = j + 1;
    //             break;
    //         }
    //         else if (dp[i] == dp[j + 1] + j - i + 1)
    //         {
    //             code.pop_back();
    //             if (ans == "")
    //                 ans = code;
    //             else
    //                 ans += " " + code;
    //             i = j + 1;
    //             break;
    //         }
    //     }
    // }

    /************ DP based Encoding Ends here ************/

    /************ Greedy Encoding Starts here ************/

    int l = 0, r = prefix.size() - 1;
    while(l<prefix.size()-1)
    {    
        if(prefix[l]==' ') l++;
        while (l < r)
        {
            string temp = prefix.substr(l, r - l + 1);
            if ((*encoding).find(temp) != (*encoding).end())
            {
                (*encoding_used)[-(*encoding)[temp]] = true;

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

    /************ Greedy Encoding Ends here ************/

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
            if (stoi(token) == 0)
            {
                flag = false;
                tokenizer >> token;
                numEncodings = stoi(token);
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
    delete transactions;
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
    int minSupport = 40;
    cout << "Min Support: " << minSupport << endl;
    std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    TreeNode *root = buildTree(transactions);
    std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    std::cout << "Time for Build Tree: " << elapsedTime.count() << " seconds" << std::endl;
    // printTree(root, 0);
    delete transactions;

    startTime = std::chrono::system_clock::now();
    TreeNode *residualTree1 = new TreeNode(-1, 0);
    bool flag = encodeTree(root, minSupport, "", residualTree1, true);
    cout << encoding->size() << endl;

    TreeNode *residualTree2 = new TreeNode(-1, 0);
    flag = encodeTree(residualTree1, minSupport / 5, "", residualTree2, false);
    cout << encoding->size() << endl;

    // TreeNode *residualTree3 = new TreeNode(-1, 0);
    // flag = encodeTree(residualTree2, minSupport / 5, "", residualTree3, false);
    // cout << encoding->size() << endl;

    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Encode Tree: " << elapsedTime.count() << " seconds" << std::endl;

    (*encoding_used).resize((*encoding).size() + 1);
    ofstream outFile(outputPath);

    startTime = std::chrono::system_clock::now();
    mineTree(root, minSupport, "", outFile);
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Mine Tree: " << elapsedTime.count() << " seconds" << std::endl;

    startTime = std::chrono::system_clock::now();
    write_mapping(outFile);
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Write Mapping: " << elapsedTime.count() << " seconds" << std::endl;
    outFile.close();

    // Remember to free the allocated memory to avoid memory leaks
    // You can create a function to delete the tree nodes recursively
    // delete encoding;
    // delete encoding_used;
    // deleteNodes(root);
    // deleteNodes(residualTree1);

    return 0;
}

float compressionRatio(string compressedFile, string originalFile)
{
    string line;
    long long sizeCf = 0, sizeOf = 0;
    // try
    // {
        ifstream cf(compressedFile);
        istringstream tokenizer;
        while (getline(cf, line))
        {
            tokenizer.clear();
            tokenizer.str(line);
            string token;
            while (tokenizer >> token)
                sizeCf++;
        }
        cf.close();

        ifstream of(originalFile);
        while (getline(of, line))
        {
            tokenizer.clear();
            tokenizer.str(line);
            string token;
            while (tokenizer >> token)
                sizeOf++;
        }
        of.close();
    // }
    // catch (exception e)
    // {
    //     cout << e.what() << ": " << line << endl;
    //     return -1;
    // }
    // cout << "sizeCf = " << sizeCf << endl;
    // cout << "sizeOf = " << sizeOf << endl;
    // return (float) sizeCf;
    return (float)sizeCf / (float)sizeOf;
}

int main(int argc, char *argv[])
{
    char *dataPath;
    char *outputPath;
    dataPath = argv[2];
    outputPath = argv[3];

    if (string(argv[1]) == "C")
    {
        compress(dataPath, outputPath);
        cout << "Compression Ratio: " << compressionRatio(outputPath, dataPath) << "\n";
    }
    // return compress(dataPath, outputPath);
    else
        return decompress(dataPath, outputPath);
}
