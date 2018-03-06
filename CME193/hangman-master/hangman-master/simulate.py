'''
This simple script is a simple template for
simulating the performance of an agent.

You can use this to see how well a custom agent performs.
'''

from hangman.hangman import Hangman
from hangman.agent import Agent

''' SET OPTIONS HERE '''
# number of simulations
nsim = 1000
# number of lives of guesser
lives = 10
# filename for wordlist
filename = 'wordsEn.txt'
# agent used
agent = Agent

''' SCRIPT '''

def parseWordFile(filename):
    wordlist = []
    with open(filename, 'r') as f:
        for line in f:
            wordlist.append(line.strip().lower())

    return wordlist

# setup simulation
wins = 0
wordlist = parseWordFile('data/' + filename)

# run games
for sim in xrange(nsim):
    a = agent(wordlist)
    h = Hangman(a, a, lives=lives, verbose=False)
    outcome = h.play()
    wins += outcome

print 'Performance: {} / {}'.format(wins, nsim)
