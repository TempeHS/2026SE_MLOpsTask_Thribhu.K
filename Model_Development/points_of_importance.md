# points of importance

really just a place for me to dump notes and observations.

---

```py
tfidf = TfidfVectorizer(
    ngram_range=(1, 2) # allows for such features like "not good"
)
```

using ngram_range improved my data from:

|   | precision | recall | f1-score | support |
| - | --------- | ------ | -------- | ------- |
| Neutral | 0.79 | 0.85 | 0.82 | 6059 |
| Negative | 0.83 | 0.80 | 0.82 | 4383 |
| Positive | 0.81 | 0.75 | 0.78 | 4032 |
| |
| Accuracy | | | 0.81 | 14474 |
| Macro Average | 0.81 | 0.80 | 0.81 | 14474 |
| Weighted Average | 0.81 | 0.81 | 0.81 | 14474 |

all the way to a:

|   | precision | recall | f1-score | support |
| - | --------- | ------ | -------- | ------- |
| Neutral | 0.91 | 0.89 | 0.90 | 6059 |
| Negative | 0.88 | 0.90 | 0.89 | 4383 |
| Positive | 0.88 | 0.88 | 0.88 | 4032 |
| |
| Accuracy | | | 0.89 | 14474 |
| Macro Average | 0.89 | 0.89 | 0.89 | 14474 |
| Weighted Average | 0.89 | 0.89 | 0.89 | 14474 |

### notes
changing the ngram_range from the default unigram [(1, 1)] to a bigram [(1, 2)] allows for features such as "not good" to exist compared to ["not", "good"]. the model/vectoriser treats "not" as a negator, while "good" would be positive. 

---
