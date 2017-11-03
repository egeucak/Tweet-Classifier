class Word:
    def __init__(self, word):
        self.word = word    #String of the word. I didn't need to call it, but what if I had to?
        self.types = []     #Count of word for different types
        self.typeProb = {}  #Probability of the word for types

    def addWord(self, count, tweetType):
        self.types.append((count,tweetType))