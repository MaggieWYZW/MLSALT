import os, codecs, sys
from nltk.stem.porter import PorterStemmer
import glob
import nltk
import re
from collections import defaultdict
import random

class MovieReviewCorpus():
    def __init__(self,stemming,pos):
        """
        initialisation of movie review corpus.

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """
        # raw movie reviews
        self.reviews=[]
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation
        self.folds=defaultdict(list)
        #self.folds={}
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # part-of-speech tags
        self.pos=pos
        # import movie reviews
        self.get_reviews()

    def get_reviews(self):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

           to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
           when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

           to use the stemmer the command is: self.stemmer.stem(token)
           
        2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """
        # TODO Q0
        # negative
        neg_path = "C:\Users\marga\Documents\MichaelmasTerm\MLSALT13\practical\mlsalt13-practical\data\\reviews\NEG\*.tag"
        pos_path = "C:\Users\marga\Documents\MichaelmasTerm\MLSALT13\practical\mlsalt13-practical\data\\reviews\POS\*.tag"
        neg_files, pos_files  = glob.glob(neg_path), glob.glob(pos_path)
        

        i = 0
        for name in neg_files:      #read from NEG folder
            neg_data = filter(lambda x: re.match(r'([a-zA-Z]+)', x), codecs.open(name, "r", "utf-8").readlines()) # only read lines starting with letters
            list = []
            if self.stemmer:
                if self.pos==True:
                    for line in neg_data:
                        list.append(tuple(self.stemmer.stem(line.split()[0]),line.split()[1]))
                else:
                    for line in neg_data:
                        list.append(self.stemmer.stem(line.split()[0]))
            else:
                if self.pos==True:
                    for line in neg_data:
                        list.append(tuple(line.split()))
                else:
                    for line in neg_data:
                        list.append(line.split()[0])

            #neg_reviews.append(("NEG", list))
            self.reviews.append(("NEG", list))

            if "cv9" in name:
                self.folds[9].append(("NEG",list))
                self.test.append(("NEG", list))
            else:
                self.train.append(("NEG", list))
            if "cv8" in name:
                self.folds[8].append(("NEG",list))
            elif "cv7" in name:
                self.folds[7].append(("NEG", list))
            elif "cv6" in name:
                self.folds[6].append(("NEG", list))
            elif "cv5" in name:
                self.folds[5].append(("NEG", list))
            elif "cv4" in name:
                self.folds[4].append(("NEG", list))
            elif "cv3" in name:
                self.folds[3].append(("NEG", list))
            elif "cv2" in name:
                self.folds[2].append(("NEG", list))
            elif "cv1" in name:
                self.folds[1].append(("NEG", list))
            elif "cv0" in name:
                self.folds[0].append(("NEG", list))
            i += 1
            #print "file No(negative):", i

        for name in pos_files:      #read from POS folder
            
            pos_data = filter(lambda x: re.match(r'([a-zA-Z]+)', x), codecs.open(name, "r", "utf-8").readlines())
            list = []
            if self.stemmer:
                if self.pos==True:
                    for line in pos_data:
                        list.append(tuple(self.stemmer.stem(line.split()[0]),line.split()[1]))
                else:
                    j=0
                    for line in pos_data:
                        list.append(self.stemmer.stem(line.split()[0]))
            else:
                if self.pos==True:
                    for line in pos_data:
                        list.append(tuple(line.split()))
                else:
                    for line in pos_data:
                        list.append(line.split()[0])

            #pos_reviews.append(("POS", list))
            self.reviews.append(("POS", list))

            if "cv9" in name:
                self.folds[9].append(("POS", list))
                self.test.append(("POS", list))
            else:
                self.train.append(("POS", list))
            if "cv8" in name:
                self.folds[8].append(("POS",list))
            elif "cv7" in name:
                self.folds[7].append(("POS", list))
            elif "cv6" in name:
                self.folds[6].append(("POS", list))
            elif "cv5" in name:
                self.folds[5].append(("POS", list))
            elif "cv4" in name:
                self.folds[4].append(("POS", list))
            elif "cv3" in name:
                self.folds[3].append(("POS", list))
            elif "cv2" in name:
                self.folds[2].append(("POS", list))
            elif "cv1" in name:
                self.folds[1].append(("POS", list))
            elif "cv0" in name:
                self.folds[0].append(("POS", list))
            i += 1
            #print "file No(positive):", i

        # Assign 10-folds reviews
        #pos_reviews = random.sample(pos_reviews, len(pos_reviews))
        #neg_reviews = random.sample(neg_reviews, len(neg_reviews))
        #for index in range(0,10):
        #    self.folds[index].extend(pos_reviews[100*index:100*(index+1)])
        #    self.folds[index].extend(neg_reviews[100*index:100*(index+1)])