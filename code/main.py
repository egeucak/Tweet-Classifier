from word import Word
import numpy as np
import string
import time
import math

'''
    wordDict= Dictionary of words, that we have in training data. It is global because we have only one training data, 
    and we are using it in a lot of functions
    
    weakProbs= Dictionary of probabilities of words with 0 frequency. Why global? I don't know, I was being lazy.
    
    train_tweets and test_tweets are self explanatory, i guess
    
    table= i thought that it will be efficient to get rid of punctuation. Guess what? It did, but in a negative way.
    But I am still gonna keep it, in case we use a larger training data, and I am sure that it will get better in that case
    If you want to see the small difference, you can uncomment #table = str.maketrans('',''). And the reason it is global is
    that I don't want to create the table every time I am using it, it is just for optimizing the code.
'''

wordDict = {}
weakProbs = {}
start = time.time()
train_tweets = np.load("train_tweets.npy")
test_tweets = np.load("validation_tweets.npy")
end = time.time()
print("Loaded tweets in {:.2f} seconds".format(end-start))

table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
#table = str.maketrans('','')

'''
    Function below normalizes tweet. In other words, it takes parses an element of data set
    This also is the part that uses the table, which I just mentioned above. 
'''
def normalize_tweet(tweet):
    type = str(tweet[0])[2:-1]
    tweet = str(tweet[1])[2:-1].translate(table).lower()
    return (type, tweet)

'''
    This is the fun part, which does a big amount of stuff. But first, I want to talk about grams. As I needed to analyze
    the difference between unigram, bigram and both of them together; I had to come up with a solution which doesn't require
    me re-writing tons of code. And that is it, I put grams I want to use in a list. If gram is 1, or 2 it does what it is supposed to do.
    But  when it comes to using both of them together, it first adds unigram then bigram to wordDict. 
'''
def trainTweets(type, tweet, grams):
    tokens = tweet.split()
    fooSet = set()
    '''
        fooset is to avoid adding same word to dictionary twice. I am sorry that I couldn't come up with a better name
    '''
    for gram in grams:
        for i in range(len(tokens)):
            if (i + gram > len(tokens)): break      #break the code, if you don't want to break the code.
            word = " ".join(tokens[i:i+gram])
            if (word in fooSet): continue
            fooSet.add(word)
            '''
                The 2 lines below are important. If I didn't write it, it wouldn't get bigger-grams or unigrams.
                Why not both?
            '''
            if (gram == 1): count = tokens.count(word)
            else: count = tweet.count(word)
            '''
                I really like using try instead of "if bla bla in bla bla". It is much faster this way in dictionaries.
                Actually, the difference was much bigger in python2, but seems like they tried fixing it in python3
                But I like doing things fast, so I didn't check if a word is in dictionary. I just put it.
            '''
            try:
                wordDict[word].addWord(count, int(type))
            except KeyError:
                inst = Word(word)
                inst.addWord(count, int(type))
                wordDict[word] = inst

'''
    I also used tf-idf, for dealing with not important words and to increase accuracy. But guess what? It didn't improve
    overall result, it decreased it. And it is because of our small training data.
'''

def idf(word):
    return math.log(1 + (len(wordDict)/len(wordDict[word].types)))

def tfidf():
    for item in wordDict.keys():
        idfN = idf(item)
        for num in range(len(wordDict[item].types)):
            wordDict[item].types[num] = (idfN * wordDict[item].types[num][0], wordDict[item].types[num][1]) #just tf-idf formula

'''
    Every word has a probability of being positive, neutral or negative. And this is the function which calculates it
'''
def findProb(word, numberOfWordsForCase, totalAmountOfWords):
    wordData = wordDict[word]
    for elm in wordData.types:
        try:    #you know; shoot first, ask questions later
            wordData.typeProb[elm[1]] += elm[0]
        except KeyError:
            wordData.typeProb[elm[1]] = elm[0]
    for case in [0, 2, 4]:      #I really hated hard coding categories, but I couldn't think of a better way at 2.00 am
        try:
            freq = wordData.typeProb[case]
        except KeyError:
            freq = 0            #If he is not telling me frequency of a case it must be 0, right?
        wordData.typeProb[case] = ((freq+1) / (numberOfWordsForCase[case] + totalAmountOfWords))    #Friendly neigborhood naive bayes


'''
    This function simply gives me number of words for every case.
'''
def getNumberOfWordsForCase():
    nums = {}
    for word in wordDict.keys():
        for type in wordDict[word].types:
            try:
                nums[type[1]] += type[0]
            except KeyError:
                nums[type[1]] = type[0]
    return nums

'''
    This part predicts case of a tweet, as it can be understood from the name
    
    I already told about the grams think, it is the same thing.
'''
def predictCase(tweet, grams, typeProb):
    tokens = tweet.split()
    predDict = {}
    '''
        As probabilities are so small, for avoiding underflow I took logarithm of every probability, and summed them,
        instead of multiplying. And here, I am taking logarithm of probability of every case
    '''
    for i in [0,2,4]:
        predDict[str(i)] = math.log(typeProb[str(i)])

    for gram in grams:
        for i in range(len(tokens)):
            if (i + gram > len(tokens)): break
            word = " ".join(tokens[i:i+gram])
            for case in [0,2,4]:
                '''
                    If we calculated probability of the word before, use it. If not, get weak probability
                '''
                try:
                    probWord = wordDict[word].typeProb[case]
                except KeyError:
                    probWord = weakProbs[case]
                predDict[str(case)] += math.log(probWord)   #Sum logarithms. (Don't forget that it sums for every gram in grams)
    #   I am summing logarithms, which gives a negative result. And for finding largest negative number,
    #   I need to have a very small number to compare to. Smaller is better. And I got myself a negative number
    #   with plenty of nines
    foo = -99999999
    for key, value in predDict.items():     #Finds the largest
        if (value > foo):
            foo = value
            result = key
    return (result)

'''
    This is for being used in situations which we don't have the probability of a word. In such case, I am calculating
    naive bayes with 0 frequency. Instead of writing (1 + 0 / (bla bla)) I just omitted unnecessary 0
'''
def setWeakProbs(numberOfWordsForCase, totalAmountOfWords):
    for i in [0,2,4]:
        weakProbs[i] = (1 / (numberOfWordsForCase[i] + totalAmountOfWords))

'''
    This is my main function. It takes grams, and tf-idf flag as input. If tf-idf is 1, it uses it.
'''
def main(grams, tf_idf):
    typeProb = {}
    for elm in train_tweets:
        type, tweet = normalize_tweet(elm)
        '''
            Try-except block below is for counting how many tweets we have for every case
        '''
        try:
            typeProb[type] += 1
        except KeyError:
            typeProb[type] = 1
        trainTweets(type, tweet, grams)     #Trains the machine, for every tweet we have. Training is not really done, but it still trains.

    if (tf_idf == 1): tfidf()   #If tf-idf flag is 1, it re-counts with tf-idf

    '''
        The part below calculates probability for every type. I could have put it in a function, but I don't feel like
        that I really need to
    '''
    total = 0
    for key in typeProb.keys():
        total += typeProb[key]
    for key in typeProb.keys():
        typeProb[key] /= float(total)

    numberOfWordsForCase = getNumberOfWordsForCase()    #Name is self-explonetary
    totalAmountOfWords = len(wordDict)                  #Name is self-explonetary

    for word in wordDict.keys():
        findProb(word, numberOfWordsForCase, totalAmountOfWords)    #Find probability of every word, for every case

    setWeakProbs(numberOfWordsForCase, totalAmountOfWords)

    '''TRAINING IS COMPLETELY DONE'''

    #Code below is really simple. so I am not explaining it
    true = 0
    false = 0
    for elm in test_tweets:
        type, tweet = normalize_tweet(elm)
        typeGuess = predictCase(tweet, grams, typeProb)
        if (typeGuess == type): true += 1
        else: false += 1

    print("Accuracy is {:.2f}%".format((true/(true+false))*100))

grams = [1]      #Which grams do you want to use? Bigram, unigram or both?
tf_idf = 0       #do you want to use tf-idf? 1 for yes

start = time.time()
main(grams, tf_idf)
end = time.time()
print("We got the results in {:.2f} seconds".format(end-start)) #With no tf-idf and unigram, it takes around 0.22 seconds. I love it when my code runs fast