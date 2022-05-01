from mrjob.job import MRJob
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import nltk
import json
import re




class Preprocess(MRJob):

    def mapper(self, _, line):
        line = line.strip() # remove leading and trailing whitespace
        review = json.loads(line)

        review_text = self.normalise(review["reviewText"])
        review_tokens_tags = self.tokenize(review["reviewText"])
        review_score = self.SentiWordNetValues(review_tokens_tags[0])
        numOfNoun = self.numberOfNouns(review_tokens_tags[1])
        numOfVerbs = self.numberOfVerbs(review_tokens_tags[1])
        numOfAdj = self.numberOfAdjectives(review_tokens_tags[1])
        numOfAdv = self.numberOfAdverbs(review_tokens_tags[1])

        result = (review["helpful"], review["overall"], review["class"], review_score, numOfNoun, numOfVerbs, numOfAdj, numOfAdv)
        yield review["_id"], result


    def normalise(self, review):
        review.lower()
        review = re.sub(r'http\S+', '', review) # removing links
        review = re.sub(r'\S*@\S*\s?', '', review) # removing emails
        review = review.translate(str.maketrans('', '', string.punctuation)) #punctuation removal
        review = review.replace("could have been", "negation")  #remove phrase and replace it with negation
        review = review.replace("hope it will be", "negation")  #remove phrase and replace it with negation        review = review.translate(str.maketrans('', '', string.punctuation)) #punctuation removal
        return(review)


    # eliminate useless words with NLTK library
    def tokenize(self, review):
        ignoring = nltk.corpus.stopwords.words("english") + [char for char in string.punctuation]
        tokens = nltk.word_tokenize(review)
        tokens = [word for word in tokens if word not in ignoring]
        #review = " ".join(tokens)
        posTags = nltk.pos_tag(tokens)
        return (tokens, posTags)

    def SentiWordNetValues(self, tokens):
        s = 0
        for token in tokens:
            sid = SentimentIntensityAnalyzer()
            s += sid.polarity_scores(token)['compound']
        return s

    def numberOfNouns(self, posTags):
        return len([word for word,pos in posTags if  'NN' in pos])

    def numberOfVerbs(self, posTags):
        return len([word for word,pos in posTags if  'VB' in pos])

    def numberOfAdjectives(self, posTags):
        return len([word for word,pos in posTags if  'JJ' in pos])

    def numberOfAdverbs(self, posTags):
        return len([word for word,pos in posTags if  'RB' in pos])


    def combiner_old(self, key, values):
        yield key, sum(values)

    def reducer_old(self, key, values):
        yield key, sum(values)


if __name__ == '__main__':
    
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('vader_lexicon')
    Preprocess.run()
