#coding: utf-8
import sys
reload(sys) 
sys.setdefaultencoding('utf-8') 
import os
import json
import jieba
import logging.config
import os.path
import math
import time
import pickle
import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec
from sklearn.cluster import KMeans
from sklearn import metrics

from proc_init import *
proc_init()
from config import *
from data_process import *
from split_word import SplitWord


logger = logging.getLogger(__name__)
setup_logging(conf_log_conf)
logger.info("load log success")
 
def formatLyric(source, target):
    data = []
    with open(source, "r") as fp:
        count = 0
        for line in fp:
            count += 1
            line = line.strip().split("]")[-1]
            #print line
            if count <= 5:
            #前5行有作者信息，删除
                continue
            elif count <= 15:
                #前10行检查双引号,大部分也是发型公司的信息
                if ":" in line or "：" in line:
                    continue
                    if line != "":
                        data.append(line)
            else:
                if line != "":
                    data.append(line)
    with open(target, "w") as fp:
        for item in data:
            print item 
            fp.write("%s\n" %(item))
def formatLyricFunc():    
    sourceDir = "/home/work/disk1/one/song_lyric/data_74"
    targetDir = "/home/work/disk1/one/song_lyric/data_lyric_74"
    for fileName in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir, fileName)
        targetFile = os.path.join(targetDir, fileName) 
        print sourceFile, targetFile
        formatLyric(sourceFile, targetFile)

def splitLyric(source,target, splitObj):
    data = []
    with open(source, "r") as fp:
        for line in fp:
            line = line.strip()
            if line == "":
                continue
            wordInfo = splitObj.splitLyric(line) 
            data.append(wordInfo)
    wordCount = {}
    for item in data:
        for key in item:
            if key in wordCount: 
                wordCount[key] += item[key]
            else:
                wordCount[key] = item[key]
    with open(target, "w") as fp:
        for key in wordCount:
            fp.write("%s\t%.5f\n" %(key, wordCount[key]))
            print key, wordCount[key]

def splitLyricWordVec(source,target, splitObj):
    data = []
    with open(source, "r") as fp:
        for line in fp:
            line = line.strip()
            if line == "":
                continue
            wordInfo = splitObj.splitLyricWordVec(line) 
            data.append(wordInfo)

    with open(target, "w") as fp:
        for line in data:
            fp.write("%s\n" %(line))

def splitFunc():
    splitObj = SplitWord()
    sourceDir = "/home/work/disk1/one/song_lyric/data_lyric/"
    targetDir = "/home/work/disk1/one/song_lyric/data_split/"
    for fileName in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir, fileName)
        targetFile = os.path.join(targetDir, fileName) 
        print sourceFile, targetFile
        splitLyric(sourceFile, targetFile, splitObj) 

def wordDictFunc():
    sourceDir = "/home/work/disk1/one/song_lyric/data_split/"
    wordSet = set()
    data = {}
    for fileName in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir, fileName)
        data[fileName] = {}
        with open(sourceFile, "r") as fp:
            for line in fp:
                arr=line.strip().split("\t")
                if len(arr) == 2:
                    data[fileName][arr[0]] = float(arr[1])
                    wordSet.add(arr[0])

    logger.info("dict len: %d" %(len(wordSet)))
    with open("./dict/word_dict.data", "w") as fp:
        for item in wordSet:
            logger.info("word dict: %s" %(item))
            fp.write("%s\n" %(item))
    logger.warn("get word dict finish")

def wordVecFunc():
    word = []
    with open("./dict/word_dict.data") as fp:
        for line in fp:
            word.append(line.strip())
    sourceDir = "/home/work/disk1/one/song_lyric/data_split/"
    targetDir = "/home/work/disk1/one/song_lyric/data_vec/"
    for fileName in os.listdir(sourceDir):
        data = {}
        vec = []
        with open(os.path.join(sourceDir, fileName), "r") as fp:
            for line in fp:
                arr = line.strip().split("\t")
                if len(arr) == 2:
                    data[arr[0]] = float(arr[1])
            for item in word:
                if item in data:
                    vec.append(data[item])
                else:
                    vec.append(0.0)
        with open(os.path.join(targetDir, fileName), "w") as fp:
            for item in vec:
                fp.write("%.5f\n" %(item))
        logger.info("count word vec %s" %(fileName))

def computeCos(va, vb):
    na = np.array(va)
    nb = np.array(vb)
    cos = np.dot(na, nb) / (np.sqrt(np.sum(np.square(na))) * np.sqrt(np.sum(np.square(nb))))
    return cos


def countCos(sid, rid):
    sourceDir = "/home/work/disk1/one/song_lyric/data_vec/"
    veca = []
    with open(os.path.join(sourceDir, sid + ".txt"), "r") as fp:
        for line in fp:
            veca.append(float(line.strip()))

    vecb = []
    with open(os.path.join(sourceDir, rid + ".txt"), "r") as fp:
        for line in fp:
            vecb.append(float(line.strip()))
    #print "data, ",veca, vecb
    return computeCos(veca, vecb) 

def findMax(sid):
    sourceDir = "/home/work/disk1/one/song_lyric/data_vec/"
    veca = []
    with open(os.path.join(sourceDir, sid + ".txt"), "r") as fp:
        for line in fp:
            veca.append(float(line.strip()))
    maxSid = ""
    maxCos = 0.0
    for fileName in os.listdir(sourceDir):
        vecb = []
        with open(os.path.join(sourceDir, fileName), "r") as fp:
            for line in fp:
                vecb.append(float(line.strip()))
        cos = computeCos(veca, vecb) 
        if cos > maxCos:
            maxSid = fileName
            maxCos = cos
        print fileName, cos
    return maxSid.split(".")[0], maxCos

def test():
    arr = []
    with open("sid.txt", "r") as fp:
        for line in fp:
            arr.append(line.strip())
    num = len(arr)
    res = []
    for i in range(num):
        res.append([0.0 for i in range(num)])

    for i in range(num):
        for j in range(i, num):
            res[i][j] = countCos(arr[i], arr[j])
    for i in range(num):
        for j in range(0,i):
            res[i][j] = res[j][i]
    for i in range(num):
        print res[i]

def wordVecSplitFunc():
    splitObj = SplitWord()
    sourceDir = "/home/work/disk1/one/song_lyric/data_lyric_74/"
    targetDir = "/home/work/disk1/one/song_lyric/data_split_word2vec_474/"
    for fileName in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir, fileName)
        targetFile = os.path.join(targetDir, fileName) 
        print sourceFile, targetFile
        splitLyricWordVec(sourceFile, targetFile, splitObj) 

def wordVecTrain():
    dataFile = "./data/lyric_line.txt"
    wordSet = set()
    data = []
    with open(dataFile, "r") as fp:
        for line in fp:
            arr = line.strip().split()
            subData = []
            for item in arr:
                word = item.decode("utf-8").strip()
                if word != "":
                    #print word
                    wordSet.add(word)
                    subData.append(word)
            if len(subData) > 0:
                data.append(subData)

    model = Word2Vec(data, min_count=1, iter=1000)
    model.save("./data/w2v.mod")
    pickle.dump(wordSet, open("./data/word_set.data", "wb"))

def countLyricVec(fileName, model, wordSet, dataDir):
    vec = None
    words = set()
    print "process ", fileName
    with open(os.path.join(dataDir, fileName), "r") as fp:
        for line in fp:
            arr = line.strip().split()
            for item in arr:
                word = item.decode("utf-8").strip()
                if word != "" and word in wordSet:
                    words.add(word)

    if len(words) > 0:
        for word in words:
            if vec is None:
                vec = model.wv[word]
            else:
                a = vec
                vec = a + model.wv[word]
    return vec

'''
res: 540389783.txt 0.75816 思美人兮
res: 604352394.txt 0.72981 如画美眷
res: 85046917.txt 0.72776 闻说

'''
def findMaxWordDoc2(sid):
    model_loaded = Word2Vec.load("./data/w2v.mod")
    print "load model finish"
    wordSet = pickle.load(open("./data/word_set.data", "rb"))
    print "load word set finish, len: %d" %(len(wordSet))
    
    dataDir = "/home/work/disk1/one/song_lyric/data_split_word2vec"
    candidates = {}
    for fileName in os.listdir(dataDir):
        vec = countLyricVec(fileName, model_loaded, wordSet, dataDir)
        if vec is not None:
            candidates[fileName] = vec

    print "load candidates finish! len: %d"  %(len(candidates))

    sFile = sid + ".txt" 
    if sFile not in candidates:
        print "sid: %s can not find" %(sid)
    else:
        sVec = candidates[sFile]
        res = []
        for item in candidates:
            if item == sFile:
                continue
            score = computeCos(sVec, candidates[item]) 
            print item, score
            res.append((item, score))
        la = sorted(res, lambda x, y: cmp(x[1], y[1]), reverse=True)
        la = la[0:100]
        for item in la:
            print "res: %s %.5f" %(item[0], item[1])
'''
255989095.txt 5.21806424116 滔滔北部湾
51814868.txt 3.91755193426 老朋友
2042051.txt 3.90272328188 弯弯的月亮
'''
def findMaxWordDoc(sid):
    model_loaded = Word2Vec.load("./data/w2v.mod")
    print "load model finish"
    dataDir = "/home/work/disk1/one/song_lyric/data_split_word2vec"
    candidates = []
    line2Doc = []
    for fileName in os.listdir(dataDir):
        with open(os.path.join(dataDir, fileName), "r") as fp:
            if fileName.split(".")[0] == sid:
                continue
            for line in fp:
                arr = line.strip().split()
                subData = []
                for item in arr:
                    word = item.decode("utf-8").strip()
                    if word != "":
                        #print word
                        subData.append(word)
                if len(subData) > 0:
                    candidates.append(subData)
                    line2Doc.append(fileName)
    print "load candidates finish! len: %d"  %(len(candidates))

    wordSet = pickle.load(open("./data/word_set.data", "rb"))
    print "load word set finish, len: %d" %(len(wordSet))
    fp = open("/home/work/disk1/one/song_lyric/data_lyric/" + sid + ".txt")
    lineSet = set()
    for line in fp:
        line = line.strip()
        print "file line: ", line
        if line in lineSet:
            continue
        else:
            lineSet.add(line)
        text = line.decode("utf-8")
        arr  = list(jieba.cut(text.strip(), cut_all=False))
        words = []
        for item in arr:
            word = item.decode("utf-8").strip()
            if word != "" and word in wordSet:
                words.append(word)
        if len(words) == 0:
            print "input len is 0, ignore"
            continue
        res = []
        index = 0
        for i in range(len(candidates)):
            candidate = candidates[i] 
            score = model_loaded.n_similarity(words, candidate)
            res.append((candidate, score, line2Doc[i]))
            index += 1
        la = sorted(res, lambda x, y:cmp(x[1], y[1]), reverse=True)
        k = 0
        for item in la:
            k += 1
            print item[1],item[2]
            for word in item[0]:
                sys.stdout.write(word)
            sys.stdout.write("\n\n")
            if k > 20:
                break

def wordVecTest():
    model_loaded = Word2Vec.load("./data/w2v.mod")
    print "load model finish"
    wordSet = pickle.load(open("./data/word_set.data", "rb"))
    print "load word set finish, len: %d" %(len(wordSet))
    dataDir = "/home/work/disk1/one/song_lyric/data_split_word2vec"
    candidates = []
    line2Doc = []
    for fileName in os.listdir(dataDir):
        with open(os.path.join(dataDir, fileName), "r") as fp:
            for line in fp:
                arr = line.strip().split()
                subData = []
                for item in arr:
                    word = item.decode("utf-8").strip()
                    if word != "" and word in wordSet:
                        #print word
                        subData.append(word)
                if len(subData) > 0:
                    candidates.append(subData)
                    line2Doc.append(fileName)
    print "load candidates finish! len: %d"  %(len(candidates))

    while True:
        text = raw_input("input sentence: ").decode("utf-8")
        arr  = list(jieba.cut(text.strip(), cut_all=False))
        words = []
        for item in arr:
            word = item.decode("utf-8").strip()
            if word != "" and word in wordSet:
                print word
                words.append(word)
        print len(words)
        if len(words) == 0:
            print "input len is 0, ignore"
            continue
        res = []
        index = 0
        for i in range(len(candidates)):
            candidate = candidates[i] 
        # print candidate
            score = model_loaded.n_similarity(words, candidate)
            res.append((candidate, score, line2Doc[i]))
            index += 1
        la = sorted(res, lambda x, y:cmp(x[1], y[1]), reverse=True)
        k = 0
        for item in la:
            k += 1
            print item[1],item[2]
            for word in item[0]:
                sys.stdout.write(word)
            sys.stdout.write("\n")
            if k > 20:
                break
        la = la[0:1000]
        doc = {}
        for item in la:
            if item[2] in doc:
                doc[item[2]] = doc[item[2]] + float(item[1])
            else:
                doc[item[2]] = float(item[1])
        lb = sorted(doc.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        for item in lb[0:10]:
            print "doc: %s count: %d" %(item[0], item[1])

def kMeanTrain():
    model_loaded = Word2Vec.load("./data/w2v.mod")
    print "load model finish"
    wordSet = pickle.load(open("./data/word_set.data", "rb"))
    print "load word set finish, len: %d" %(len(wordSet))
    
    dataDir = "/home/work/disk1/one/song_lyric/data_split_word2vec"
    #dataDir = "/home/work/disk1/one/song_lyric/test"
    candidates = []
    files = []
    for fileName in os.listdir(dataDir):
        vec = countLyricVec(fileName, model_loaded, wordSet, dataDir)
        if vec is not None:
            candidates.append(vec)
            files.append(fileName)

    print "load candidates finish! len: %d"  %(len(candidates))
    #print candidates
    X = np.array(candidates) 
    '''
    for clusterNum in [5,10,15,20,25]:
        model = KMeans(n_clusters=clusterNum)
        y_pred = model.fit_predict(X)
        newScore =  metrics.calinski_harabaz_score(X, y_pred)
        print "score",newScore
    sys.exit(0)
    '''
    clusterNum = 5
    model = KMeans(n_clusters=clusterNum)
    y_pred = model.fit_predict(X)
    newScore =  metrics.calinski_harabaz_score(X, y_pred)
    print "score",newScore

    for i in range(len(X)):
        print "cluster: %s %s" %(files[i], y_pred[i])
    order_centroids = model.cluster_centers_
    print order_centroids
    def findMax(all, vec):
        la = []
        for i in range(len(all)):
            score = computeCos(all[i], vec)
            la.append((i,score))
        lb = sorted(la, lambda x, y: cmp(x[1], y[1]), reverse=True)
        return lb[0:10]

    for i in range(clusterNum):
        maxIndex = findMax(candidates, order_centroids[i])
        for j in range(len(maxIndex)):
            print "cluster %d: maxIndex: %d file: %s socre: %.5f" %(i, j, files[maxIndex[j][0]], float(maxIndex[j][1]))
#
def findChannelSongs(filePath):
    chData = {}
    for line in open(filePath, "r"):
        chData[line.strip()] = []
    print "get channel songs len: %d" %(len(chData))
    model_loaded = Word2Vec.load("./data/w2v.mod")
    print "load model finish"
    wordSet = pickle.load(open("./data/word_set.data", "rb"))
    print "load word set finish, len: %d" %(len(wordSet))
    
    dataDir = "/home/work/disk1/one/song_lyric/data_split_word2vec"
    candidates = {}
    for fileName in os.listdir(dataDir):
        vec = countLyricVec(fileName, model_loaded, wordSet, dataDir)
        if vec is not None:
            candidates[fileName] = vec

    print "load candidates finish! len: %d"  %(len(candidates))

    simData = {}
    for sid in chData:
        sFile = sid + ".txt" 
        if sFile not in candidates:
            print "sid: %s can not find" %(sid)
        else:
            sVec = candidates[sFile]
            res = []
            for item in candidates:
                if item == sFile:
                    continue
                score = computeCos(sVec, candidates[item]) 
                print item, score
                res.append((item, score))
            la = sorted(res, lambda x, y: cmp(x[1], y[1]), reverse=True)
            la = la[0:10]
            for item in la:
                rid = item[0].split(".")[0]
                if rid in chData:
                    print "ignore song : %s by in channel" %(rid)
                print "chsong: %s res: %s %.5f" %(sid, item[0], item[1])
                if item[0] in simData:
                    simData[item[0]] += item[1]
                else:
                    simData[item[0]] = item[1]
    la = sorted(simData.items(), lambda x, y: cmp(x[1], y[1]), reverse=True) 
    la = la[0:100]
    for item in la:
        print "final res: %s %.5f" %(item[0], item[1])

if __name__ == "__main__":
    if sys.argv[1] == "format":
        formatLyricFunc()
    elif sys.argv[1] == "split":
        splitFunc()
    elif sys.argv[1] == "worddict":
        wordDictFunc()
    elif sys.argv[1] == "wordvec":
        wordVecFunc()
    elif sys.argv[1] == "wordvec_split":
        wordVecSplitFunc()
    elif sys.argv[1] == "wordvec_train":
        wordVecTrain()
    elif sys.argv[1] == "kmean_train":
        kMeanTrain()
    elif sys.argv[1] == "channel":
        findChannelSongs("42_chid.txt")        
    elif sys.argv[1] == "test":
        #test()
        #findMax("209713")
        #splitObj = SplitWord()
        #splitLyricWordVec("/home/work/disk1/one/song_lyric/data_lyric/209713.txt", "./209713_split.txt", splitObj) 
        wordVecTest()
        #findMaxWordDoc2("209713")
        #findChannelSongs("74_chid.txt")        

