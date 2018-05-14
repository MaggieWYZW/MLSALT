from Analysis import Evaluation

class SentimentLexicon(Evaluation):
    def __init__(self):
        """
        read in lexicon database and store in self.lexicon
        """
        # if multiple entries take last entry by default
        self.lexicon=dict([[l.split()[2].split("=")[1],l.split()] for l in open("data/sent_lexicon","r")])
        self.predictions=[]


    def classify(self,reviews,threshold,magnitude):
        """
        classify movie reviews using self.lexicon.
        self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["priorpolarity=negative","type=strongsubj"].
        explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
        store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for.
                          experiment for good threshold values.
        @type threshold: integer

        @type magnitude: use magnitude information from self.lexicon?
        @param magnitude: boolean
        """

        # reset predictions
        self.predictions=[]
        # TODO Q0
        for label, review in reviews:
            positive = 0
            negative = 0
            if magnitude == True:
                for word in review:
                    if word in self.lexicon:
                        if self.lexicon[word][5] == "priorpolarity=positive":
                            positive += 2
                            if self.lexicon[word][0] == "type=strongsubj":
                                positive += 1
                            elif self.lexicon[word][0] == "type=weaksubj":
                                positive -= 1
                        elif self.lexicon[word][5] == "priorpolarity=negative":
                            negative += 2
                            if self.lexicon[word][0] == "type=strongsubj":
                                negative += 1
                            elif self.lexicon[word][0] == "type=weaksubj":
                                negative -= 1
            else:
                for word in review:
                    if word in self.lexicon:
                        if self.lexicon[word][5] == "priorpolarity=positive":
                            positive += 1
                        elif self.lexicon[word][5] == "priorpolarity=negative":
                            negative += 1

            if positive - negative > threshold:
                result = "POS"
            else:
                result = "NEG"

            if label == result:
                #print "review prediction index", reviews_index
                self.predictions.append("+")
            else:
                self.predictions.append("-")
                #print "review prediction index", reviews_index
                