# Corpus 
A quick start container for NLP. 

## Quick Use 
```python
>>> import pandas as pd 
>>> import Corpus as cp

>>> # create sample text files
>>> docs = [
    'This is a long and tedius document', 
    'This is also a document 432d.', 
    'A pig flew into my house to eat', 
    'I roasted and ate a pig for dinner'
]

>>> # store documents in rows of a dataframe
>>> df = pd.DataFrame()
>>> df['text'] = docs 
>>> df = df.set_index(pd.Index([1, 2, 3, 4]))
>>> df

                                 text
1  This is a long and tedius document
2       This is also a document 432d.
3     A pig flew into my house to eat
4  I roasted and ate a pig for dinner

>>> # fit the corpus 
>>> corpus = cp.Corpus(df['text'], stopwords=True, stem=True)
>>> corpus.df 

                             raw_text              stemmed                       ngrams                   bag_ngrams
1  This is a long and tedius document  long tediu document    long_tediu,tediu_document  [1, 1, 0, 0, 0, 0, 0, 0, 0]
2       This is also a document 432d.        also document                also_document  [0, 0, 1, 0, 0, 0, 0, 0, 0]
3     A pig flew into my house to eat    pig flew hous eat  pig_flew,flew_hous,hous_eat  [0, 0, 0, 1, 1, 1, 0, 0, 0]
4  I roasted and ate a pig for dinner  roast at pig dinner   roast_at,at_pig,pig_dinner  [0, 0, 0, 0, 0, 0, 1, 1, 1]

>>> corpus.show_lda_topics(ngram=2, n_topics=3, n_words=5)

############ LDA TOPICS ############
0: 0.173*"also_document" + 0.143*"hous_eat" + 0.141*"flew_hous" + 0.136*"tediu_document" + 0.094*"pig_dinner"
1: 0.171*"pig_flew" + 0.141*"pig_dinner" + 0.136*"at_pig" + 0.105*"roast_at" + 0.103*"flew_hous"
2: 0.167*"roast_at" + 0.162*"long_tediu" + 0.137*"at_pig" + 0.102*"tediu_document" + 0.102*"hous_eat"

>>> corpus.get_ngram_ct('pig_flew', 3)
1
>>> corpus.show_top_ngrams(ngram=1, n=3)

############ TOP NGRAMS ############
document: 2
pig: 2
long: 1
```

## Options 
In addition to the quick use cases above, you can also do the following things: 
1. Export top ngrams and topics to a `.csv`
   - `corpus.export_top_ngrams(ngram=2, n=10)`
   - `corpus.export_lda_topics(ngram=2, n_topics=10, n_words=5)`
1. View and save a wordcloud
   - `corpus.wordcloud(ngram=2)`
   - `corpus.export_wordcloud(ngram=2)`