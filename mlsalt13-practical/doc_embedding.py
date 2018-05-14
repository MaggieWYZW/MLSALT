# question 8.0
from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec
import gensim
import glob
import os.path
import requests
import tarfile
import sys
import codecs
import smart_open
import re
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
import time

# retrieve corpus
corpus=MovieReviewCorpus(stemming=False,pos=False)

# use sign test for all significance testing
signTest=SignTest()
svmlight_dir = "C:\Users\marga\Documents\MichaelmasTerm\MLSALT13\practical\svm_light\\"


print "--- using document embeddings ---"
dirname = "imdb"

# convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
	norm_text=[]
	for all in text:
		temp_text = all.lower()
		temp_text = temp_text.replace("<br />", " ") #replace breaks with spaces
		control_chars = [".", "'", '"', ',', '(', ')', '!', '?', ';', ':', "\\", "/"]
		for char in control_chars: #pad punctuation with spaces on both sides
			temp_text = temp_text.replace(char, ' ')
		norm_text.append(temp_text)
	return norm_text

print "---------- Cleaning up dataset ---------"
alldata=[]
folders=["train/pos","train/neg","test/pos","test/neg"]#,"train/unsup"]
for item in folders:
	temp=[]
	output = item.replace('/','-')+'.txt'
	txt_files = glob.glob(os.path.join(dirname, item, "*.txt"))
	i=0
	for txt in txt_files:   # read all text files in one folder
		label=re.findall("_(\d+).txt",txt)[0].decode("utf-8")+" "  #get the document tag from filename
		with smart_open.smart_open(txt,"rb") as t:
			t_clean=t.read().decode("utf-8")
		temp.append(label + t_clean)
		i+=1
		print i
	temp_norm = normalize_text(temp)
	#with smart_open.smart_open(os.path.join(dirname, output), "wb") as n:
	#	n.writelines("%s\n" % item.encode("utf-8") for item in temp_norm)
	alldata.extend(temp_norm)

files=["train-pos.txt","train-neg.txt","test-pos.txt","test-neg.txt"]
alldata=[]
for file in files:
	for line in codecs.open(os.path.join(dirname,file),"rb",encoding="utf-8"):
		if re.match(r"[\d+]", line):  #this method ignore part of the data 
			alldata.append(line)

with smart_open.smart_open(os.path.join(dirname, "alldata.txt"), "wb") as f:
	for idx, line in enumerate(alldata):
		num_line=u"{0}_*{1}\n".format(idx,line)
		f.write(num_line.encode("utf-8"))
		print idx
print "--------------- Finish dataset processing ----------------"

print "------------ Preparing data for training --------------"

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # Will hold all docs in original order
#with codecs.open('imdb/alldata.txt', encoding='utf-8') as alldata:
    #for line_no, line in enumerate(alldata):
for line_no, data in enumerate(alldata):
    tokens = gensim.utils.to_unicode(data).split()
    words = tokens[1:]
    tags = tokens[0]   # 'tags = [tokens[0]]' would also work at extra memory cost
    split = ['train', 'test', 'extra', 'extra'][line_no//25000]  # 25k train, 25k test, 25k extra
    sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
    alldocs.append(SentimentDocument(words, tags, split, sentiment))
    print "line number:",line_no

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # For reshuffling per pass
#train_docs += test_docs # I want to use all the data as training data
print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))


print "----------------- Train Doc2vec Models ----------------"

from gensim.models import Doc2Vec

cores = multiprocessing.cpu_count()
#assert gensim.models.doc2vec.FAST_VERSION > -1  #"This will be painfully slow otherwise"
simple_models = [
    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/ average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

# Speed up setup by sharing results of the 1st model's vocabulary scan
simple_models[0].build_vocab(alldocs)  # PV-DM w/ concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)

############ Concatenating Two Models
#print "------------ Concatenating Two Models -----------------"
#from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
#models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
#models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])
print "number of models to train:" , len(models_by_name)
#print "---------- Train SVM with Doc2vec -----------"
print "-------- MODEL TRAINING ---------------"
from random import shuffle
import datetime
alpha, min_alpha, passes = (0.025, 0.001, 6)
alpha_delta = (alpha - min_alpha) / passes
for epoch in range(passes):
    shuffle(doc_list)
    print "iteration:", (epoch+1)
    for name, train_model in models_by_name.items():
        print name
        print("START %s" % datetime.datetime.now())
        train_model.alpha, train_model.min_alpha = alpha, alpha
        train_model.train(doc_list, total_examples=len(doc_list), epochs=1)
        #train_model.save(model_name)
        print("END %s" % str(datetime.datetime.now()))

i=0
for name, model in models_by_name.items():
    i+=1
    model_name = "model_%d" % i
    model.save(model_name)

from gensim.models import Doc2Vec
print "------- loading models ---------"
dmc_model= models_by_name.items()[0][1]#Doc2Vec.load("model_1")
cbow_model=models_by_name.items()[1][1]#Doc2Vec.load("model_2")
dmm_model=models_by_name.items()[2][1]#Doc2Vec.load("model_3")

corpus=MovieReviewCorpus(stemming=False,pos=False)
print "----------- Distributive Memory + SVM -----------"
SVM=SVMDoc2Vec(model=dmc_model, svmlight_dir=svmlight_dir)
SVM.crossValidate(corpus)
print "Average Accuracy: %.2f" % SVM.getAccuracy()
print "Std. Dev: %.2f" % SVM.getStdDeviation()

print "--------- Distributed Bag of Words + SVM ------------"
SVM=SVMDoc2Vec(model=cbow_model, svmlight_dir=svmlight_dir)
SVM.crossValidate(corpus)
print "Average Accuracy: %.2f" % SVM.getAccuracy()
print "Std. Dev: %.2f" % SVM.getStdDeviation()

print "---------- Average Distributive Memmory + SVM -----------"
SVM=SVMDoc2Vec(model=dmm_model, svmlight_dir=svmlight_dir)
SVM.crossValidate(corpus)
print "Average Accuracy: %.2f" % SVM.getAccuracy()
print "Std. Dev: %.2f" % SVM.getStdDeviation()
#print "------------- DBOW & DMM + SVM -------------"
#SVM=SVMDoc2Vec(model=simple_models[0], svmlight_dir=svmlight_dir)

#print "--------------- DBOW & DMC + SVM -------------"
#SVM=SVMDoc2Vec(model=simple_models[0], svmlight_dir=svmlight_dir)
'''