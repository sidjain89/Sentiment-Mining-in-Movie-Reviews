import numpy as np
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from nltk.tokenize import RegexpTokenizer
import datetime as dt
import random as rd


pos_path = "pos_corpus.txt"
neg_path = "neg_corpus.txt"
pos_wt = 1
neg_wt = -1
pos_text = ""
neg_text = ""
posFolderPath = "D:\\IDS 566\\movie_reviews\\pos\\"
negFolderPath = "D:\\IDS 566\\movie_reviews\\neg\\"
posList = list()
negList = list()

def ReadContents(path):
    f = open(pos_path, "r")
    text = f.read().decode("utf-8",errors="ignore")
    text = re.sub('[^a-zA-Z]+', " ", text.lower()) 
    f.close()

    # tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    print("removing stopwords:" + str(dt.datetime.now()))

    # remove stop words

    filtered = tokens

    #filtered = [w for w in tokens if not w in stopwords.words('english')]

    print("removed stopwords:" + str(dt.datetime.now()))
    return filtered

#end def

def ExtractWordCount(wordList):
    word_dic = dict()
    lst = list()
    lst = wordList
    uniq_words = np.unique(wordList)

    for word in uniq_words:
        word_dic[word] = -1 * lst.count(word)

    return word_dic
#end def

def SaveScores(path, text):
    f = open(path, "w")
    f.write(text)
    f.close()
#end def

def  ParseFiles(folderPath):
    fileList = os.listdir(folderPath)

    rd.shuffle(fileList)

    #fileList = fileList[:700]

    i = len(fileList)

    txt = ""

    for fileName in fileList:
        file = open(folderPath + fileName, "r")
        text = file.read().decode("utf-8",errors="ignore")
        text = re.sub('[^a-zA-Z]+', " ", text.lower())        
        file.close()

        print(fileName)

        #global neg_text

        #neg_text += text + "\n "

        txt = text.replace("\r\n", " ")
        txt = txt.replace("\n", " ")
        
        global negList

        negList.append(txt)
        negList.append("\n")


    f = open("pos_corpus_1000_new.txt", "w")
    f.writelines(negList)
    f.close()
#end def

#pos_list = ReadContents(pos_path)
#pos_dict = ExtractWordCount(pos_list)

#for k,v in pos_dict.iteritems():
#    pos_text +=  "\n" + k + "," + str(v) 

#SaveScores("pos_scores.csv", pos_text)

#print("Done with positive!")

#neg_list = ReadContents(neg_path)
#neg_dict = ExtractWordCount(neg_list)

#for k,v in neg_dict.iteritems():
#    neg_text +=  "\n" + k + "," + str(v) 

#SaveScores("neg_scores.csv", neg_text)


#print("Done with negative!")

#ParseFiles(negFolderPath)

ParseFiles(posFolderPath)

print("Done!")



