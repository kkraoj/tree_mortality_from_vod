class Hangman:
    '''
    Hangman class implement the game dynamics of Hangman

    To run, it requires a Chooser agent, a Guesser agent, and a list of words
    '''
    def __init__(self, chooser, guesser, lives=10,
                 verbose=True, debug=False):
        '''
        Setup of the game
        '''
        # the agents
        self.chooser = chooser
        self.guesser = guesser

        self.lives = lives  # number of lives for guesser

        self.verbose = verbose
        self.debug = debug

        self.guesses = ''

    def play(self):
        '''
        The function that runs one game
        '''
        self.printStart()

        # Let the chooser pick the word
        self.getWord()

        # bookkeeping
        self.nguesses = 0
        self.win = False

        # Turns of guesser
        while self.lives > 0:
            # Some output
            self.printRound()

            # Get the guess from the guesser
            self.getGuess()
            self.nguesses += 1

            # if no more characters are missing, we found the word
            if len(self.missing_chars) == 0:
                self.win = True
                break

        # Print the outcome of the game
        self.printOutcome()
        # Return whether the guesser won or lost
        return self.win

    def getWord(self):
        '''
        Function to query the chooser for a word
        '''
        # Ask the chooser to pick a word from the wordlist
        self.word = self.chooser.getWord()

        # Convert the word into a set of characters: all these characters need
        # to be guessed.
        self.missing_chars = {ch for ch in self.word}

        # To assist in debugging
        if self.debug:
            print 'Hidden word: {}'.format(self.word)

    def hideWord(self):
        '''Return the hidden word where we only show characters that have
        # been guessed correctly'''
        output = ''

        for ch in self.word:
            # if it is still missing, put a dot
            if ch in self.missing_chars:
                output += '. '
            # otherwise, put the character
            else:
                output += ch + ' '
        return output

    def getGuess(self):
        '''
        Get and process a guess from the guesser
        '''
        guess = self.guesser.getGuess(self.hideWord())

        # save guess for possible analysis
        self.guesses += guess
        if self.verbose:
            print 'Guess: {}'.format(guess)

        if guess in self.missing_chars:
            # if the guess is one of the missing characters,
            # it is no longer missing!
            self.missing_chars.remove(guess)
        else:
            # otherwise, the guess was incorrect and the guesser loses a life.
            self.lives -= 1

    def printRound(self):
        '''
        Output to be printed before every round
        '''
        if self.verbose:
            print '-' * 20
            print 'Round: {} \t Lives left: {}'.format(self.nguesses,
                                                       self.lives)
            print 'Current word: {}'.format(self.hideWord())

    def printStart(self):
        '''
        Output to be printed before the game starts
        '''
        if self.verbose:
            print '='*20
            print 'HANGMAN GAME'
            print 'lives: {}'.format(self.lives)
            print '='*20

    def printOutcome(self):
        '''
        Output to be printed after the game has finished
        '''
        if self.verbose:
            print '=' * 20
            if self.win:
                print 'The guesser wins!'
                print 'Lives left: {}'.format(self.lives)
            else:
                print 'The chooser wins!'

            print 'Guesses used: {}'.format(self.nguesses)
            print 'The word the chooser was looking for: {}'.format(self.word)

    def summary(self):
        '''
        Return a summary of the game, could be useful to analyse performance
        '''
        return {'win': self.win, 'nguesses': self.nguesses,
                'nlives': self.lives, 'word': self.word,
                'guesses': self.guesses, 'missing': self.missing_chars}
