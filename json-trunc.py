import json
from gensim import corpora, models, similarities
from collections import Counter

# list up all the reviews in NV
f = open("review_phoenix.json")
reviews = json.load(f)
f.close()

f = open('review_phoenix_trunc.json', 'w')
f.write('[');
limit = 20000;
for i in range(limit):
	json.dump(reviews[i], f);
	if (i != limit -1):
		f.write(',');
f.write(']');
f.close()
