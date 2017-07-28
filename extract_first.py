import sys

with open(sys.argv[1], 'r') as f :
    for line in f:
        line = line.strip()
        tokens = line.split(' ', 1)
        print(tokens[1])
