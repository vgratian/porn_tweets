
# porn_tweets

A simple Python script to calculate cosine similarity between tweets. The program takes as input an file with (preferably large number of) tweets.
We used an archive with 1M tweets from 2016. A small subset of this collection is separated as `benchmark` and the rest of the collection is compared to the benchmark and ranked by decreasing cosine similarity.

This code was a homework assignment for the course Information Retrieval and Text Mining at the University of Stuttgart. I thank my coursemate Jyothsna Vijayendra for explaining me the computation of cosine similarity.

## benchmark & results

We chose as our benchmark 3 tweets with adult content. We were curious what percentage of the top ranking tweets would also be of adult content. And if yes, what characteristics would divide them from the lower ranking tweets. The resulting top 100 tweets all unambiguously belonged to the pornographic genre.

We noticed that those were not only tweets which contained relatively high number of words from the benchmark terms, but they furthermore contained matching words that were less frequent in the collection (i.e. words with lower tf-idf score). For example `sex` had a document frequency (df) of 4792, while `cumshots` only 102, therefore the latter had a higher tf-idf score and tweets containing it would rank higher.

In short, tweets which contained a larger number of matching words and especially words that were less frequent in the collection, had a higher cosine similarity to the benchmark.

The ranking of 1M tweets required 124.15 seconds on an XPS Dell laptop with i5 Intel processor and 8GB memory.

## tweet data

In order to run the program, a 'tweets' collection file must be in the folder `data`. In our case the tweet file contained five tab-separated fields in the following format:

```
DATESTAMP TWEET_ID  USER_ACCOUNT  USER_NAME TWEET_TEXT
```

Only `tweet_id` and `tweet_text` are used in this program.

## stopwords
The list of 119 stop words were copied and adapted from [Stanford Core NLP](https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt)
which is under the GNU General Public license.
