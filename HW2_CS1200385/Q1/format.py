import sys

with open(sys.argv[1], 'r') as file:
    graph_lines = file.readlines()
    char_2_num = {}
    curr = 0
    num = 0
    i = 0
    while i < len(graph_lines):
        line = graph_lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('#'):
            cnt = int(graph_lines[i+1].strip())
            print("t # {}".format(num))
            for j in range(cnt):
                char = graph_lines[i+j+2].strip()
                if char not in char_2_num:
                    char_2_num[char] = curr
                    curr += 1
                print("v {} {}".format(j, char_2_num[char]))
            num += 1
            i += cnt + 2
        else:
            cnt = int(graph_lines[i].strip())
            for j in range(cnt):
                print("e {}".format(graph_lines[i+j+1].strip()))
            i += cnt + 1
