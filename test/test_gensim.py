import pandas as pd 
import gensim 
import sys 

sys.path.append('../')
import Corpus as cp

docs = [
    'This is a long and tedius document', 
    'This is also a document 432d.', 
    'A pig flew into my house to eat dinner', 
    'I roasted and ate a pig for dinner'
]

d = pd.DataFrame()
d['text'] = docs 
d = d.set_index(pd.Index([1, 2, 3, 4]))

corpus = cp.Corpus(d['text'])

