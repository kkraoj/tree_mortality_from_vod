'''
This script is useful for running a game of Hangman from the command line
'''

import argparse

from hangman.hangman import Hangman
from hangman.agent import Agent, HumanAgent


def parseWordFile(filename):
    wordlist = []
    with open(filename, 'r') as f:
        for line in f:
            wordlist.append(line.strip().lower())

    return wordlist


def parseAgentOptions(option, wordlist):
    if option == 'a':
        return Agent(wordlist)
    if option == 'h':
        return HumanAgent(wordlist)
    raise ValueError('option {} not available')


agentChoices = ['a', 'h']

parser = argparse.ArgumentParser(description='Hangman game.')

parser.add_argument('-c', '--chooser', help='Set chooser agent',
                    choices=agentChoices, default='a')
parser.add_argument('-g', '--guesser', help='Set guesser agent',
                    choices=agentChoices, default='h')
parser.add_argument('-w', '--wordlist', help='Path to wordlist to use',
                    default='wordsEn.txt')
parser.add_argument('-v', '--verbose', help='Verbose output',
                    action="store_true")

args = parser.parse_args()

# ensure verbose is on when using human player
if args.guesser == 'h':
    args.verbose = True

wordlist = parseWordFile('data/' + args.wordlist)

chooser = parseAgentOptions(args.chooser, wordlist)
guesser = parseAgentOptions(args.guesser, wordlist)

h = Hangman(chooser, guesser, verbose=args.verbose)

outcome = h.play()
