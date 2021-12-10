# Utilities
import re
import pickle
# DATA 
import pandas as pd
import numpy as np
# NLP
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwd = stopwords.words('spanish')

col = pickle.load(open('../model/IchigoModel.columns.v1.pckl', 'rb'))
str_col = [str(c) for c in col]

def transform(text) -> list:
  arr = np.zeros(len(col)).tolist()
  arr[0] = text

  tokens = [ w for w in word_tokenize(arr[0]) if w not in stopwd and re.match('\w', w) ]
  bigrams = list(nltk.bigrams(tokens))

  for idx, ele in enumerate(col):
    if ele in tokens: 
      arr[idx] = 1
    elif len(bigrams):
      for b in bigrams:
        if ele[0] == b[0] and ele[1] == b[1]:
          arr[idx] = 1

  df = pd.DataFrame([arr], columns=str_col)
  df.drop(columns=['question', 'intent'], inplace=True)
  
  return df
