import pickle
from nltk.tokenize import RegexpTokenizer
import re
import os
import random

folder = "D:\\IDS 566\\movie_reviews\\pos\\"
filesToClassify = ["test_review_1.txt", "test_review_2.txt"]

filesToClassify = os.listdir(folder)

random.shuffle(filesToClassify)
filesToClassify = filesToClassify[:10]

posScore = float()
negScore = float()
posDict = dict()
negDict = dict()

f = open("posDict.pickle", "rb")
posDict = pickle.load(f)
f.close()

#print(type(posDict))

f = open("negDict.pickle", "rb")
negDict = pickle.load(f)
f.close()

for file in filesToClassify:
    f = open(folder + file, "r")
    text = f.read()
    text = re.sub('[^a-zA-Z]+', " ", text.lower())  
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    for token in  tokens:
        for key in posDict.keys():
            val = str(key)
            if(val.count(token) > 0 or str(token).count(key) > 0):
                posScore += posDict[key]

        for key in negDict.keys():
            val = str(key)
            if(val.count(token) > 0 or str(token).count(key) > 0):
                negScore += negDict[key]
    #end for


    msg = ""

    posScore = posScore / 100
    negScore = negScore / 100

    if(posScore > 120 and negScore < 110):
        msg = "| This is a positive review!"

    elif(negScore > 0 and posScore > 0):
        if(posScore/ negScore > 1.1):
            msg = "| This is a positive review!"

        if(posScore/ negScore < 1.1):
            msg = "| This is a negative review!"

        if(posScore/ negScore == 1.1):
            msg = "| This is a neutral review!"

    else:
        msg = "| This is a neutral review!"

    print("For file: " + file)  
    print("positivity = " + str(posScore) + " and " + "negativity = " + str(negScore) + msg)
# end for

inp = input("Done!")
### end script ###





