# Back-Propagation Neural Networks
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random
import string
import sys
import json
import datetime
import argparse

parser = argparse.ArgumentParser(description='Train neural network.')
parser.add_argument('-s', action="store_true",
                    help='save neural network to file.')
parser.add_argument('--no-test', action="store_true", default=False,
                    help='test and show results.')
parser.add_argument('-l', action="store", nargs=1, type=str,
                    help='load neural network from file.')
parser.add_argument('samples', action="store", nargs="?", type=str,
                    help='load neural network from file.')
args = parser.parse_args()

def nowAsString():
    retval = "{:%y%m%d%H%M%S}".format(datetime.datetime.today())
    retval += ".nn.json" #Filename extension
    return retval

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh + 1 # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.001, M=0.001):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)

    def save(self):
        with open(nowAsString(), "w") as text_file:
            text_file.write(json.dumps({
                'wo': self.wo,
                'wi': self.wi
            }, indent=2))

    @staticmethod
    def load(filename):
        content = "";
        with open(filename, 'r') as content_file:
            content = content_file.read()
        source = json.loads(content)

        self = NN(0, 0, 0);

        # number of input, hidden, and output nodes
        self.ni = len(source["wi"])
        self.nh = len(source["wo"])
        self.no = len(source["wo"][0])

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = source["wi"]
        self.wo = source["wo"]

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        return self

class player:
    def __init__(self):
        content = "";
        with open('samples', 'r') as content_file:
            content = content_file.read()
        pat = eval(content);
        print pat
        # create a network with two input, two hidden, and one output nodes
        self.mynet = NN(43, 10, 7)
        # train it with some patterns
        self.mynet.train(pat)

    def play(self, turn, board):
        width=7
        height=6
        myboard=[]
        for x in range(height-1,-1, -1):
            for y in range(0,width):
                myboard.append(board[y][x])

        myboard.append(turn)

        move=[]
        move.append(myboard)
        move.append([0,0,0,0,0,0,0])

        outputActivations = self.mynet.update(move)
        bestOption = outputActivations.index(max(outputActivations))
        while myboard[bestOption] != 0:
            outputActivations[bestOption] = -1
            bestOption = outputActivations.index(max(outputActivations))
        return bestOption

def demo():
    # Train (default) or load
    if args.l != None:
        if args.no_test or args.samples == None:
            print "Not doing anything. When loading neural network, provide a samples file to test against."
            sys.exit();
        nn = NN.load(args.l[0])
        content = "";
        with open(args.samples, 'r') as content_file:
            content = content_file.read()
        pat = eval(content)
        nn.test(pat)
    elif args.samples != None:
        nn = NN(43, 10, 7)
        content = "";
        with open(args.samples, 'r') as content_file:
            content = content_file.read()
        pat = eval(content)
        nn.train(pat)
        if not args.no_test:
            nn.test(pat)
        if args.s:
            nn.save()
    else:
        print "Must train or load neural network..."
        parser.print_help()
        sys.exit()

if __name__ == '__main__':
    demo()
