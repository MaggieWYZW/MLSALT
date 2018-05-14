import os
from subprocess import call
from nltk.util import ngrams
from Analysis import Evaluation
from collections import Counter
import numpy as np
from operator import itemgetter


class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: booleanp

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        # set of features for classifier
        self.vocabulary=set() # what does set do?
        # prior probability
        self.prior={}
        # conditional probablility
        self.condProb={}
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]

    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)
        return self.vocabulary

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for token in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(token)==2 and self.discard_closed_class:
                if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
            else:
                text.append(token)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety 
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """

        # TODO Q1
        self.vocabulary = set()
        self.prior = {}     # prior & condprob are both dictionaries
        self.condProb={}
        #extract vocabulary - nonrepeating words in both POS and NEG reviews
        self.vocabulary = self.extractVocabulary(reviews)
        print "number of features(size of vocabulary):", len(self.vocabulary)
        N_D = len(reviews)
        # class document count
        N_POS = 0
        N_NEG = 0
        text_pos = []
        text_neg = []
        for label, review in reviews:
            if label == "POS":
                N_POS+=1
                text_pos.extend(self.extractReviewTokens(review))
            if label == "NEG":
                N_NEG+=1
                #print "Number of Negative reviews:", N_NEG
                text_neg.extend(self.extractReviewTokens(review))
            
        self.prior["POS"] = float(N_POS)/N_D # positive prior
        self.prior["NEG"] = float(N_NEG)/N_D # negative prior

        N_t_pos = Counter(text_pos)
        N_t_neg = Counter(text_neg)
        self.condProb["POS"] = {}
        self.condProb["NEG"] = {}

        # TODO Q2 (use switch for smoothing from self.smoothing)
        if self.smoothing == True:
            for term in self.vocabulary:
                self.condProb["POS"][term] = float(N_t_pos[term]+1)/(len(text_pos)+len(self.vocabulary))
                self.condProb["NEG"][term] = float(N_t_neg[term]+1)/(len(text_neg)+len(self.vocabulary))
        elif self.smoothing == False:
            for term in self.vocabulary:
                if N_t_pos[term]:
                    self.condProb["POS"][term]= float(N_t_pos[term])/len(text_pos)
                if N_t_neg[term]:
                    self.condProb["NEG"][term] = float(N_t_neg[term])/len(text_neg)

        return self.vocabulary, self.prior, self.condProb

        
        
    def test(self,reviews):
        """
        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.
        
        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q1
        self.predictions = []
        i = 0
        for label, review in reviews:
            log_prob_pos = np.log(self.prior["POS"])
            log_prob_neg = np.log(self.prior["NEG"])
            tokens = self.extractReviewTokens(review)

            for token in tokens:
                if token in self.condProb["POS"]:
                    log_prob_pos += np.log(self.condProb["POS"][token])
                if token in self.condProb["NEG"]:
                    log_prob_neg += np.log(self.condProb["NEG"][token])

            if log_prob_pos>log_prob_neg:
                prediction = "POS"
            else:
                prediction = "NEG"

            if label == prediction:
                self.predictions.append("+")
            else:
                self.predictions.append("-")

            ## for track of program
            i += 1
            #print "review prediction No:", i

        return self.predictions


class SVM(Evaluation):
    """
    general svm class to be extended by text-based classifiers.
    """
    def __init__(self,svmlight_dir):
        self.predictions=[]
        self.svmlight_dir=svmlight_dir

    def writeFeatureFile(self,data,filename):
        """
        write local file in svmlight data format.
        see http://svmlight.joachims.org/ for description of data format.

        @param data: input data
        @type data: list of (string, list) tuples where string is the label and list are features in (id, value) tuples

        @param filename: name of file to write
        @type filename: string
        """
      # TODO Q6.0
        write_file = open(filename, "w")
        for label, content in data:
            write_file.write(label+"\t")
            write_file.write("\t".join('%s:%s' %feature for feature in content)+"\n")
        write_file.close()
                
    def train(self,train_data):
        """
        train svm 

        @param train_data: training data 
        @type train_data: list of (string, list) tuples corresponding to (label, content)
        """
        # function to determine features in training set. to be implemented by child 
        self.getFeatures(train_data)
        # function to find vectors (feature, value pairs). to be implemented by child
        train_vectors=self.getVectors(train_data)
        self.writeFeatureFile(train_vectors,"train.data")
        # train SVM model
        call([self.svmlight_dir+"svm_learn","train.data","svm_model"],stdout=open(os.devnull,'wb'))
        #print "SVM feature size:", len(self.vocabulary)

    def test(self,test_data):
        """
        test svm 

        @param test_data: test data 
        @type test_data: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q6.1
        test_vectors=self.getVectors(test_data)
        self.writeFeatureFile(test_vectors, "test.data")
        call([self.svmlight_dir+"svm_classify","test.data","svm_model","svm_predictions"], stdout=open(os.devnull, 'wb'))
        self.predictions=[]
        predictions=[float(line) for line in open("svm_predictions","r").readlines()]
        for i in range(0,len(predictions)):
            if test_vectors[i][0]=="1" and predictions[i]>0:
                self.predictions.append("+")
            elif test_vectors[i][0]=="-1" and predictions[i]<0:
                self.predictions.append("+")
            else:
                self.predictions.append("-")


class SVMText(SVM):
    def __init__(self,bigrams,trigrams,svmlight_dir,discard_closed_class):
        """ 
        initialisation of SVMText object

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        SVM.__init__(self,svmlight_dir)
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class

    def getFeatures(self,reviews):
        """
        determine features from training reviews and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # reset for each training iteration
        self.vocabulary=set()
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review): 
                self.vocabulary.add(token)
        # using dictionary of vocabulary:index for constant order
        # features for SVMLight are stored as: (feature id, feature value)
        # using index+1 as a feature id cannot be 0 for SVMLight
        self.vocabulary={token:index+1 for index,token in enumerate(self.vocabulary)}

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for term in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(term)==2 and self.discard_closed_class:
                if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
            else:
                text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(term)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(term)
        return text

    def getVectors(self,reviews):
        """
        get vectors for svmlight from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of (string, list) tuples where string is the label ("1"/"-1") and list
                 contains the features in svmlight format e.g. ("1",[(1, 0.04), (2, 4.0), ...])
                 svmlight feature format is: (id, value) and id must be > 0.
        """
        # TODO Q6.1
        vectors=[]
        for label, review in reviews:
            list = []
            if label=="POS":
                svm_label = "1"
                feature_value=Counter(review)
                for word in feature_value:
                    if word in self.vocabulary:
                        list.append(tuple((self.vocabulary[word], feature_value[word])))
                        list=sorted(list,key=itemgetter(0))
            else:
                svm_label= "-1"
                feature_value=Counter(review)
                for word in feature_value:
                    if word in self.vocabulary:
                        list.append(tuple((self.vocabulary[word], feature_value[word])))
                        list=sorted(list,key=itemgetter(0))
            vectors.append(tuple((svm_label,list)))
        return vectors