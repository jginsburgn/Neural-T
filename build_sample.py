import sys
import string

arg = []
winner = -1

for line in sys.stdin:
    arg.append(line)
# print arg

if "USELESS" in arg[len(arg) - 1]:
    # print "fuckoff"
    sys.exit()
else:
    winner = int(arg[len(arg) - 1])
    # print winner,

for ind in range(len(arg) - 2):
    arg[ind] = arg[ind][:arg[ind].find("]") + 1] + arg[ind + 1][arg[ind + 1].find("]") + 1:]

for ind in range(len(arg) - 2):
    if winner==2:
        if ind%2==0:
            arg[ind] = string.replace(arg[ind], "3", "1")
            arg[ind] = string.replace(arg[ind], "]", ", 2]", 1)
        else:
            arg[ind] = string.replace(arg[ind], "3", "-1")
            arg[ind] = string.replace(arg[ind], "]", ", 1]", 1)
    else:
        if ind%2==0:
            arg[ind] = string.replace(arg[ind], "3", "-1")
            arg[ind] = string.replace(arg[ind], "]", ", 2]", 1)
        else:
            arg[ind] = string.replace(arg[ind], "3", "1")
            arg[ind] = string.replace(arg[ind], "]", ", 1]", 1)

del arg[len(arg) - 1]
del arg[len(arg) - 1]
for ind in range(len(arg) - 1):
    print "[" + arg[ind][:len(arg[ind]) - 2] + "],"
