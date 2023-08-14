#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>

class TreeNode
{
public:
    int item;
    int count;
    TreeNode *parent;
    TreeNode *next;
    std::unordered_map<int, TreeNode *> children;

    TreeNode(int _item, int _count, TreeNode *_parent = nullptr)
        : item(_item), count(_count), parent(_parent) {}
};

void buildHeaderTable(const std::vector<std::vector<int>> &transactions, std::unordered_map<int, int> &frequency)
{
    for (const auto &transaction : transactions)
        for (int item : transaction)
            frequency[item]++;
}

void insertTransaction(TreeNode *node, const std::vector<int> &transaction, std::unordered_map<int, TreeNode *> &headerTable)
{
    if (transaction.empty())
        return;

    int item = transaction[0];
    TreeNode *child;
    if (node->children.find(item) != node->children.end())
    {
        child = node->children[item];
        child->count++;
    }
    else
    {
        child = new TreeNode(item, 1, node);
        node->children[item] = child;
        if (headerTable[item] == nullptr)
            headerTable[item] = child;
        else
        {
            while (headerTable[item]->next != nullptr)
                headerTable[item] = headerTable[item]->next;
            headerTable[item]->next = child;
        }
    }

    insertTransaction(child, std::vector<int>(transaction.begin() + 1, transaction.end()), headerTable);
}

TreeNode *buildFPTree(const std::vector<std::vector<int>> &transactions, int minSupport)
{
    std::unordered_map<int, int> frequency;
    buildHeaderTable(transactions, frequency);

    std::vector<int> frequentItems;
    for (const auto &entry : frequency)
        if (entry.second >= minSupport)
            frequentItems.push_back(entry.first);

    std::sort(frequentItems.begin(), frequentItems.end(), [&](int a, int b)
              { return frequency[a] > frequency[b]; });

    std::unordered_map<int, TreeNode *> headerTable;
    TreeNode *root = new TreeNode(-1, 0);
    for (const auto &transaction : transactions)
    {
        std::vector<int> filteredTransaction;
        for (int item : transaction)
            if (std::find(frequentItems.begin(), frequentItems.end(), item) != frequentItems.end())
                filteredTransaction.push_back(item);

        std::sort(filteredTransaction.begin(), filteredTransaction.end(), [&](int a, int b)
                  { return frequency[a] > frequency[b] || (frequency[a] == frequency[b] && a < b); });
        insertTransaction(root, filteredTransaction, headerTable);
    }

    return root;
}

int main(int argc, char *argv[])
{
    // Reading transactions
    bool compress = true;
    char *dataPath;
    char *outputPath;
    if (argv[1] == "D")
        compress = false;
    dataPath = argv[2];
    outputPath = argv[3];

    std::vector<std::vector<int>> transactions;

    std::ifstream inputFile(dataPath);
    if (!inputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(inputFile, line))
    {
        std::vector<int> tokens;
        std::istringstream tokenizer(line);
        std::string token;

        while (tokenizer >> token)
            tokens.push_back(std::stoi(token));

        transactions.push_back(tokens);
    }

    int minSupport = 2;
    TreeNode *root = buildFPTree(transactions, minSupport);

    // Print the FP-tree (not complete visualization)
    // You can create a function for better visualization
    std::cout << "FP-tree structure:" << std::endl;
    std::cout << "Root" << std::endl;
    for (const auto &childPair : root->children)
    {
        TreeNode *child = childPair.second;
        std::cout << "  " << child->item << " (" << child->count << ")" << std::endl;
    }

    // Remember to free the allocated memory to avoid memory leaks
    // You can create a function to delete the tree nodes recursively
    delete root;

    return 0;
}