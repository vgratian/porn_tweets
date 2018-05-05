import string, timeit
from math import log10, sqrt

"""
This program compares tweets from a collection to a benchmark (a subset of the
collection, in this case: three tweets with adult content) and sorts them by
cosine similiarity. The top tanking 100 tweets are printed.

The default 3 benchmark tweets are displayed in the dictionary below
(tweets are normalized and URL's have been moved).
"""

global benchmark, fp_tweets, fp_stopwords, stopwords, collection_size
global terms, terms_tf, terms_df, terms_idf, terms_tf_idf, matched_tweets, top

benchmark = {
    '766419051145035777': 'First anal sex in the kitchen #amateur #anal #blond #blowjob #brunette #cumshots',
    '776913092257148928': '#women sleeping sex porn actor nude',
    '779008212288823296': '#dark pussy hair teens lesbian movies'
    }
collection_size = 0 # total number of tweets
stopwords = [] # frequent words to be ignored
terms = {} # dict with terms as keys and count as values
terms_tf = {}
terms_df = {}
terms_idf = {}
terms_tf_idf = {}
matched_tweets = {} # final result, format: {'tweet' : cos_sim score}
top = [] # will contain top 100 results from matched tweets
fp_tweets = 'data/tweets'
fp_stopwords = 'data/stopwords'

def run():
    """
    Main routine. Will call all responsable functions to calculate tf-idf
    scores and cosine similarity for the benchmark terms and the tweets.

    Benchmark terms, top 100 results and runtime will be printed.
    """
    # start counting runtime
    start = timeit.default_timer()

    # populate global variable stopwords
    import_stopwords()

    # extract terms and calulate term-frequencies in benchmark
    merge_benchmark()
    global terms_tf
    terms_tf = {term:terms[term]/len(terms) for term in terms.keys()}

    # calculate tf-idf scores for benchmark
    get_idf_scores()
    global terms_tf_idf
    terms_tf_idf = {term:terms_tf[term]*terms_idf[term] for term in terms}

    # print terms and df-s
    print('{} terms in benchmark. Printing terms with document frequency (df):'
        .format(len(terms)))
    for term in terms.keys():
        print('({}) {}'.format(terms_df[term], term))
    print('Ranking {} tweets by cosine similarity.'.format(collection_size))

    # calculate tf-idfs for tweets and find cosine similiarity to benchmark
    find_tweets_sim()

    # sort matched tweets and slice top 100 results
    top = sorted(matched_tweets.items(), key=lambda x:x[1], reverse=True)[:100]

    # end runtime counting
    stop = timeit.default_timer()
    runtime = stop - start

    # print runtime and top 100 results
    print('Found {} matching tweets in {} seconds.'.format
        (len(matched_tweets), round(runtime, 2)))
    print('Printing 100 tweets with highest cosine similarity...')
    for tweet in top:
        print('({})\t{}'.format(tweet[1], tweet[0]))


def merge_benchmark():
    """
    Will merge benchmark tweets and extract unique terms.
    """
    global terms, benchmark
    for tweet in benchmark:
        tokens = tokenize_tweet(benchmark[tweet]) # returns list of tokens
        for word in tokens:
            if word not in terms.keys():
                terms[word] = 1
            else:
                terms[word] += 1


def find_tweets_sim():
    """
    Calculates tf-idf score for tweets in collection and compare to benchmark.
    """
    global terms, terms_idf, matched_tweets
    tweets = read_tweets()
    for tweet in tweets:
        tweet = tweet.split('\t')
        if len(tweet) == 5: # Skip tweets in wrong format or missing data
            tweet_text = tweet[4]
            tweet_tokens = tokenize_tweet(tweet_text)
            tweet_id = tweet[1]
            if tweet_tokens and tweet_id not in benchmark.keys(): # Skip tweets with no tokens
                tf_idf_list = {}
                at_least_one_match = False
                for term in terms.keys():
                    tf = 0
                    for token in tweet_tokens:
                        if term == token:
                            tf += 1
                            at_least_one_match = True
                    tf = tf/len(tweet_tokens)
                    tf_idf = tf * terms_idf[term]
                    tf_idf_list[term] = tf_idf
                if at_least_one_match:
                    cos_sim = get_cos_sim(tf_idf_list)
                    matched_tweets[tweet_text] = cos_sim


def get_idf_scores():
    """
    Computes document frequency for terms (i.e. number of tweets containing the
    term). Converts them to idf scores.
    Also computes the collection size. Both variables are globally changed.
    """
    global terms, terms_idf, terms_df, collection_size
    tweets = read_tweets()

    # First get document frequency (df)
    terms_df = {term:0 for term in terms}
    collection_size = 0
    for tweet in tweets:
        tweet = tweet.split('\t')
        if len(tweet) == 5:
            tweet_tokens = tokenize_tweet(tweet[4])
            tweet_id = tweet[1]
            if tweet_id not in benchmark.keys():
                collection_size += 1
                for term in terms:
                    if term in tweet_tokens:
                        terms_df[term] += 1
    # Get idf from df
    for term in terms_df:
        df = terms_df[term]
        idf = 1 + log10( collection_size / df ) if df != 0 else 0
        terms_idf[term] = idf


def get_cos_sim(tweet_vectors):
    dot_prod = get_dot_prod(tweet_vectors)
    eucl_len_prod = get_eucl_len_prod(tweet_vectors)

    return dot_prod / eucl_len_prod


def get_dot_prod(tweet_vectors):
    global terms_tf_idf
    dot_prod = 0
    for term in tweet_vectors.keys():
        dot_prod += tweet_vectors[term] * terms_tf_idf[term]
    return dot_prod


def get_eucl_len_prod(tweet_vectors):
    global terms_tf_idf
    eucl_len0 = 0
    eucl_len1 = 0
    for term in tweet_vectors.keys():
        eucl_len0 += pow(tweet_vectors[term], 2)
        eucl_len1 += pow(terms_tf_idf[term], 2)
    return sqrt(eucl_len0) * sqrt(eucl_len1)


def import_stopwords():
    """
    Import stop words from file into global variable stopwords
    """
    global fp_stopwords, stopwords
    try:
        f = open(fp_stopwords, 'r')
    except:
        print('Can\'t read stopwords file')
        raise IOError
    else:
        stopwords = f.read().splitlines()
    finally:
        f.close()


def tokenize_tweet(tweet):
    """
    Minimal tokenization interface. Splits string into words by space. Removes
    punctuation, URLs, removes stop words. Returns only alphabetical string
    words, lowercased.
    """
    global stopwords
    removepunct = str.maketrans('', '', string.punctuation)
    dirty_tokens = tweet.split()
    clean_tokens = []
    for token in dirty_tokens:
        token = token.translate(removepunct).lower()
        if token.isalpha() and token not in stopwords and 'http' not in token:
            clean_tokens.append(token)
    return clean_tokens


def read_tweets():
    """
    Generator that returns tweets from file one by one
    """
    global fp_tweets
    try:
        f = open(fp_tweets, 'r')
    except:
        print('Can\'t read tweet file')
        raise IOError
    else:
        for line in f.readlines():
            yield line
    finally:
        f.close()


if __name__ == '__main__':
    run()
