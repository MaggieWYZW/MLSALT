from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec

#import nltk

# retrieve corpus
#corpus=MovieReviewCorpus(stemming=False,pos=False) # original
corpus=MovieReviewCorpus(stemming=False,pos=False)

# use sign test for all significance testing
signTest=SignTest()

# location of svmlight binaries 
# TODO: change this to your local installation
#svmlight_dir="/path/to/svmlight/binaries/"
svmlight_dir = "C:\Users\marga\Documents\MichaelmasTerm\MLSALT13\practical\svm_light\\"
'''
print "--- classifying reviews using sentiment lexicon  ---"
# read in lexicon
lexicon=SentimentLexicon()
# on average there are more positive than negative words per review (~7.13 more positive than negative per review)
# to take this bias into account will use threshold (roughly the bias itself) to make it harder to classify as positive
threshold=8
# question 0.1
lexicon.classify(corpus.reviews,threshold,magnitude=False)
token_preds=lexicon.predictions
print "token-only results: %.2f" % lexicon.getAccuracy()

lexicon.classify(corpus.reviews,threshold,magnitude=True)
magnitude_preds=lexicon.predictions
print "magnitude results: %.2f" % lexicon.getAccuracy()

# question 0.2
p_value=signTest.getSignificance(token_preds,magnitude_preds)
print "magnitude lexicon results are",("significant" if p_value < 0.05 else "not significant"),"with respect to token-only","(p=%.8f)" % p_value



# question 1.0
print "--- classifying reviews using Naive Bayes on held-out test set ---"
NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
non_smoothed_preds=NB.predictions # store predictions from classifier
print "Accuracy without smoothing: %.2f" % NB.getAccuracy()
#print "--- Cross-validation ---"
#NB.crossValidate(corpus)
#print "Average Accuracy without smoothing: %.2f" % NB.getAccuracy()


# question 2.0
# use smoothing
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
smoothed_preds=NB.predictions
num_non_stemmed_features=len(NB.vocabulary)
#NB.crossValidate(corpus)
#smoothed_preds=NB.predictions
# saving this for use later
print "Accuracy using smoothing: %.2f" % NB.getAccuracy()

# question 2.1
# see if smoothing significantly improves results
p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
print "results using smoothing are",("significant" if p_value < 0.05 else "not significant"),"with respect to no smoothing","(p=%.8f)" % p_value

# question 3.0
print "--- non-stemming corpus ---"
#print "--- classifying reviews using 10-fold cross-evaluation ---"
# using previous instantiated object
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
#NB.crossValidate(corpus)
NB.train(corpus.train)
NB.test(corpus.test)
non_stemmed_preds=NB.predictions
num_corpus_features=len(NB.vocabulary)
# using cross-eval for smoothed predictions from now on
print "Accuracy: %.2f" % NB.getAccuracy()
#print "Std. Dev: %.2f" % NB.getStdDeviation()
print "feature size:", num_corpus_features

# question 4.0
print "--- stemming corpus ---"
# retrieve corpus with tokenized text and stemming (using porter)
stemmed_corpus=MovieReviewCorpus(stemming=True,pos=False)
######################################
NB.train(stemmed_corpus.train)
NB.test(stemmed_corpus.test)
num_stemmed_corpus_features=len(NB.vocabulary)
###################################
#print "--- cross-validating NB using stemming ---"
#NB.crossValidate(stemmed_corpus)
stemmed_preds=NB.predictions             # prediction size change after cross validation, it becomes 10 times.
print "Accuracy: %.2f" % NB.getAccuracy()
#print "Std. Dev: %.2f" % NB.getStdDeviation()
print "feature size:", num_stemmed_corpus_features

# TODO Q4.1
# see if stemming significantly improves results on smoothed NB
p_value=signTest.getSignificance(non_stemmed_preds,stemmed_preds)
print "results using stemming are",("significant" if p_value < 0.05 else "not significant"),"with respect to no stemming","(p=%.8f)" % p_value
'''

# TODO Q4.2
#print "--- determining the number of features before/after stemming ---"
#print "Corpus features before Stemming:", num_non_stemmed_features
#print "Corpus features after Stemming:", num_stemmed_corpus_features

# question Q5.0
# cross-validate model using smoothing and bigrams
#print "--- cross-validating naive bayes using smoothing and bigrams ---"
print "--- Uni-gram + Bi-gram ---"
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False)
#NB.crossValidate(corpus)
NB.train(corpus.train)
NB.test(corpus.test)
print " ----- Non-stemmed corpus -----"
smoothed_and_bigram_preds=NB.predictions
print "Average Accuracy: %.2f" % NB.getAccuracy()
#print "Std. Dev: %.2f" % NB.getStdDeviation()
#NB.crossValidate(stemmed_corpus)
print "--- Uni-gram only ---"
# see if bigrams significantly improves results on smoothed NB only
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
#NB.crossValidate(corpus)
NB.train(corpus.train)
NB.test(corpus.test)
smoothed_preds=NB.predictions
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
print "results using smoothing and bigrams are",("significant" if p_value < 0.05 else "not significant"),"with respect to smoothing only","(p=%.8f)" % p_value

# TODO Q5.1
# cross-validate model using smoothing and trigrams
#print "--- cross-validating naive bayes using smoothing and trigrams ---"
print " --- Uni-gram + Tri-gram ---"
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=True,discard_closed_class=False)
#NB.crossValidate(corpus)
NB.train(corpus.train)
NB.test(corpus.test)
smoothed_and_trigram_preds=NB.predictions
print "Averate Accuracy: %.2f" % NB.getAccuracy()
#print "Std. Dev: %.2f" % NB.getStdDeviation()
# see if trigrams significantly improve results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_trigram_preds)
print "results using smoothing and trigrams are",("significant" if p_value < 0.05 else "not significant"),"with respect to smoothing only","(p=%.8f)" % p_value
'''

# TODO Q6 and 6.1
print "--- classifying reviews using SVM 10-fold cross-eval ---"
SVM=SVMText(bigrams=False,trigrams=False,svmlight_dir=svmlight_dir,discard_closed_class=False)
#SVM.train(corpus.train)
#SVM.test(corpus.test)
SVM.crossValidate(corpus)
svm_preds=SVM.predictions
print "Average Accuracy: %.2f" % SVM.getAccuracy()
print "Std. Dev: %.2f" % SVM.getStdDeviation()
#p_value=signTest.getSignificance(smoothed_preds,svm_preds) # non-stemming data
#print "results using SVM are",("significant" if p_value < 0.05 else "not significant"),"with respect to NB","(p=%.8f)" % p_value


# TODO Q7
print "--- adding in POS information to corpus ---"
pos_corpus=MovieReviewCorpus(stemming=False,pos=True)

print "--- training svm on word+pos features ----"
SVM.crossValidate(pos_corpus)
svm_pos_preds=SVM.predictions
print "Average Accuracy: %.2f" % SVM.getAccuracy()
print "Std. Dev: %.2f" % SVM.getStdDeviation()
p_value=signTest.getSignificance(svm_preds,svm_pos_preds) # non-stemming data
print "results using POS tag are",("significant" if p_value < 0.05 else "not significant"),"with respect to SVM only","(p=%.8f)" % p_value

print "--- training svm discarding closed-class words ---"
SVM=SVMText(bigrams=False,trigrams=False,svmlight_dir=svmlight_dir,discard_closed_class=True)
SVM.crossValidate(pos_corpus)
svm_pos_discard_class_preds=SVM.predictions
print "Average Accuracy: %.2f" % SVM.getAccuracy()
print "Std. Dev: %.2f" % SVM.getStdDeviation()
p_value=signTest.getSignificance(svm_preds,svm_pos_discard_class_preds) # non-stemming data
print "results discarding closed class are",("significant" if p_value < 0.05 else "not significant"),"with respect to SVM only","(p=%.8f)" % p_value
'''
'''
