import pickle

posDict = dict()
negDict = dict()
lines = list()


print("reading most important features....")
f = open("1000_words.csv","r")
lines = f.readlines()
f.close()

for line in lines:
    arr = str(line).split(",")
    val = float(str(arr[2]).replace("\n",""))
    if arr[1] == "pos":
        posDict[arr[0]] = val
    else:
        negDict[arr[0]] = val
# end for

pLen = len(posDict) + 1
nLen = len(negDict) + 1

# normalize weights

#alpha = [(word, (weight + 1)/pLen) for (word,weight) in posDict.items()]
#beta = [(word, (weight + 1)/nLen) for (word,weight) in negDict.items()]

print("Normalizing weights....")

for key in posDict.keys():
    posDict[key] = (posDict[key] + 1) * 100 / pLen


for key in negDict.keys():
    negDict[key] = (negDict[key] + 1) * 100 / nLen


#print(type(alpha))

print("creating pickle...")

f = open("posDict.pickle", "wb")
pickle.dump(posDict, f)
f.close()

f = open("negDict.pickle", "wb")
pickle.dump(negDict, f)
f.close()

inp = input("Done!")




