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
    long long cc_count = 0;
    map<int, TreeNode *> children = map<int, TreeNode *>();

    TreeNode(int _item = -1, long long _count = 0, map<int, TreeNode *> _children = map<int, TreeNode *>())
        : item(_item), count(_count), copy_count(_count), cc_count(_count) {}
};

map<string, int> *encoding = new map<string, int>();
vector<int> *encoding_used = new vector<int>();
int value = -2, num_transactions = 0;

int fileSize = 2;    // file size 1 means small and medium files, 0 means large file, 2 for semi-large files

TreeNode *buildTree(vector<vector<int>> *transactions)
{
    unordered_map<int, long long> *frequency = new unordered_map<int, long long>();
    for (const auto &transaction : *transactions)
        for (int item : transaction)
            (*frequency)[item]++;

    TreeNode *root = new TreeNode(-1, 0);

    for (auto &transaction : *transactions)
    {
        sort(transaction.begin(), transaction.end(), [&](int a, int b)
             { return (*frequency)[a] > (*frequency)[b] || ((*frequency)[a] == (*frequency)[b] && a < b); });

        TreeNode *node = root;
        for (int item : transaction)
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
                TreeNode *child = new TreeNode(item, 1);
                node->children[item] = child;
                node = child;
            }
        }
    }
    delete frequency;
    return root;
}

TreeNode *buildTree_copy(vector<vector<int>> *transactions)
{
    unordered_map<int, long long> *frequency = new unordered_map<int, long long>();
    for (const auto &transaction : *transactions)
        for (int item : transaction)
            (*frequency)[item]++;

    TreeNode *root = new TreeNode(-1, 0);
    for (auto &transaction : *transactions)
    {
        sort(transaction.begin(), transaction.end(), [&](int a, int b)
             { return (*frequency)[a] < (*frequency)[b] || ((*frequency)[a] == (*frequency)[b] && a > b); });

        TreeNode *node = root;
        for (int item : transaction)
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
                TreeNode *child = new TreeNode(item, 1);
                node->children[item] = child;
                node = child;
            }
        }
    }
    delete frequency;
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
            to_node->children[child->item] = new TreeNode(child->item, child->count, child->children);
    }
}

bool encodeTree(TreeNode *node, int min_support, string prefix, TreeNode *newTreeRoot, bool buildResidual)
{
    if (node->children.size() == 0)
        return false;

    if (node->item >= 0)
        prefix += to_string(node->item) + " ";

    int node_copy_count = node->copy_count;

    for (const auto childPair : node->children)
    {
        TreeNode *child = childPair.second;
        if (node_copy_count < min_support && node->item >= 0)
        {
            if (!buildResidual)
                break;

            if (newTreeRoot->children.find(child->item) != newTreeRoot->children.end())
                mergeTree(child, newTreeRoot->children[child->item]);
            else
                newTreeRoot->children[child->item] = new TreeNode(child->item, child->count, child->children);

            continue;
        }
        if (encodeTree(child, min_support, prefix, newTreeRoot, buildResidual))
            node_copy_count -= child->count;
    }
    if (node_copy_count >= min_support && (*encoding).find(prefix) == (*encoding).end() && node->item >= 0 && count(prefix.begin(), prefix.end(), ' ') > 1)
    {
        (*encoding)[prefix] = value--;
        return true;
    }
    return false;
}

int encodeTree_copy(TreeNode *node, int min_support, string prefix)
{
    if (node->children.size() == 0)
        return -1;

    if (node->item >= 0)
        prefix = to_string(node->item) + " " + prefix;

    int node_copy_count = node->copy_count;

    for (const auto childPair : node->children)
    {
        TreeNode *child = childPair.second;
        if (node_copy_count < min_support && node->item >= 0)
            break;
        int child_count = encodeTree_copy(child, min_support, prefix);
        if (child_count != -1)
            node_copy_count -= child_count;
    }
    if (node_copy_count >= min_support && (*encoding).find(prefix) == (*encoding).end() && node->item >= 0 && count(prefix.begin(), prefix.end(), ' ') > 1)
    {
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
    outFile << "-1 " << size << endl;

    for (auto e : (*encoding)){
        if(fileSize == 1){
            if ((*encoding_used)[-e.second] > 1)
            {
                if (count(e.first.begin(), e.first.end(), ' ') == 1 and (*encoding_used)[-e.second] == 2)
                    continue;
                outFile << e.second << "\n"
                        << e.first << endl;
            }
        }
        else{
            outFile << e.second << "\n"
                        << e.first << endl;
        }
    }
}

// uses O(n) greedy algo
void processTransaction_Big_files(string prefix, long long freq, ofstream &outFile)
{
    string ans = "";
    if (freq > 1)
        ans = "0" + to_string(freq);

    int l = 0, r = prefix.size() - 1;
    while (l < r)
    {
        string temp = prefix.substr(l, r - l + 1);
        if ((*encoding).find(temp) != (*encoding).end())
        {
            (*encoding_used)[-(*encoding)[temp]] = true;

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
        // cout << ans << ",\n";
    }
    if (l < prefix.size() - 1)
        if (ans == "")
            ans = prefix.substr(l, prefix.size() - l - 1);
        else
            ans += " " + prefix.substr(l, prefix.size() - l - 1);

    if (outFile.is_open())
        outFile << ans << "\n";
    else
        std::cerr << "Failed to open the output file." << std::endl;
}

void processTransaction_1(string prefix, long long freq, ofstream &outFile)
{

    /************ DP based Encoding Starts here ************/
    // string ans = "";

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
    while (l < prefix.size() - 1)
    {
        if (prefix[l] == ' ')
            l++;
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
        int end = l + 1;
        while (end < prefix.size() and prefix[end] != ' ')
            end++;
        l = end + 1;
        r = prefix.size() - 1;
    }

    /************ Greedy Encoding Ends here ************/
}

void processTransaction_2(string prefix, long long freq, ofstream &outFile)
{
    string ans = "";
    if (freq > 1)
        ans = "0" + to_string(freq);
    /************ Greedy Encoding Starts here ************/

    int l = 0, r = prefix.size() - 1;
    while (l < prefix.size() - 1)
    {
        if (prefix[l] == ' ')
            l++;
        while (l < r)
        {
            string temp = prefix.substr(l, r - l + 1);
            if ((*encoding).find(temp) != (*encoding).end() and (*encoding_used)[-(*encoding)[temp]] > 1)
            {
                if (count(temp.begin(), temp.end(), ' ') == 1 and (*encoding_used)[-(*encoding)[temp]] == 2)
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
        int end = l + 1;
        while (end < prefix.size() and prefix[end] != ' ')
            end++;
        ans += " " + prefix.substr(l, end - l);
        l = end + 1;
        r = prefix.size() - 1;
    }
    if (ans[0] == ' ')
        ans = ans.substr(1);

    /************ Greedy Encoding Ends here ************/

    if (outFile.is_open())
        outFile << ans << "\n";
    else
        std::cerr << "Failed to open the output file." << std::endl;
}

void mineTree_1(TreeNode *node, string prefix, ofstream &outfile)
{
    if (node == nullptr)
        return;
    if (node->item >= 0)
        prefix += to_string(node->item) + " ";

    for (const auto &childPair : node->children)
    {
        TreeNode *child = childPair.second;
        node->count -= child->count;
        mineTree_1(child, prefix, outfile);
    }

    if (node->count > 0){
        if(fileSize == 1)
            processTransaction_1(prefix, node->count, outfile);
        else 
            processTransaction_Big_files(prefix, node->count, outfile);
    }

}

void mineTree_2(TreeNode *node, string prefix, ofstream &outfile)
{
    if (node == nullptr)
        return;
    if (node->item >= 0)
        prefix += to_string(node->item) + " ";

    for (const auto &childPair : node->children)
    {
        TreeNode *child = childPair.second;
        node->cc_count -= child->cc_count;
        mineTree_2(child, prefix, outfile);
    }

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
            if (token == "")
                continue;
            if (stoi(token) == -1)
            {
                flag = false;
                tokenizer >> token;
                numEncodings = stoi(token);
            }
            else if (token[0] == '0')
            {
                int x = 0;
                while (line[x] != ' ')
                    x++;
                x++;
                for (int i = stoi(token); i; i--)
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
    int minSupport = 20;
    cout << "Min Support: " << minSupport << endl;
    std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    TreeNode *root = buildTree(transactions);
    std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    std::cout << "Time for Build Tree 1: " << elapsedTime.count() << " seconds" << std::endl;

    if(fileSize == 1)
    {
        startTime = std::chrono::system_clock::now();
        TreeNode *root_copy = buildTree_copy(transactions);
        endTime = std::chrono::system_clock::now();
        elapsedTime = endTime - startTime;
        std::cout << "Time for Build Tree 2: " << elapsedTime.count() << " seconds" << std::endl;

        startTime = std::chrono::system_clock::now();
        int flag_copy = encodeTree_copy(root_copy, max(5, minSupport / 4), "");
        endTime = std::chrono::system_clock::now();
        elapsedTime = endTime - startTime;
        std::cout << "Time for Encode Tree 2: " << elapsedTime.count() << " seconds" << std::endl;

        deleteNodes(root_copy);
    }

    delete transactions;

    startTime = std::chrono::system_clock::now();
    if(fileSize >= 1)
    {
        TreeNode *residualTree1 = new TreeNode(-1, 0);
        bool flag = encodeTree(root, minSupport, "", residualTree1, true);
        cout << encoding->size() << endl;
        
        TreeNode *residualTree2 = new TreeNode(-1, 0);
        flag = encodeTree(residualTree1, minSupport / 2, "", residualTree2, true);
        cout << encoding->size() << endl;

        TreeNode *residualTree3 = new TreeNode(-1, 0);
        flag = encodeTree(residualTree2, minSupport / 4, "", residualTree3, false);
        cout << encoding->size() << endl;

        deleteNodes(residualTree1);
        deleteNodes(residualTree2);
    }
    else{
        TreeNode *residualTree1 = new TreeNode(-1, 0);
        bool flag = encodeTree(root, minSupport, "", residualTree1, false);
        cout << encoding->size() << endl;
    }
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Encode Tree: " << elapsedTime.count() << " seconds" << std::endl;

    (*encoding_used).resize((*encoding).size() + 1);
    ofstream outFile(outputPath);

    startTime = std::chrono::system_clock::now();
    mineTree_1(root, "", outFile);
    endTime = std::chrono::system_clock::now();
    elapsedTime = endTime - startTime;
    std::cout << "Time for Mine Tree 1: " << elapsedTime.count() << " seconds" << std::endl;

    if(fileSize == 1)
    {
        startTime = std::chrono::system_clock::now();
        mineTree_2(root, "", outFile);
        endTime = std::chrono::system_clock::now();
        elapsedTime = endTime - startTime;
        std::cout << "Time for Mine Tree 2: " << elapsedTime.count() << " seconds" << std::endl;
    }

    write_mapping(outFile);
    outFile.close();

    // Remember to free the allocated memory to avoid memory leaks
    // You can create a function to delete the tree nodes recursively
    delete encoding;
    delete encoding_used;
    deleteNodes(root);

    return 0;
}

float compressionRatio(string compressedFile, string originalFile)
{
    string line;
    long long sizeCf = 0, sizeOf = 0;
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
