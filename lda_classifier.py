import json
import pprint
from gensim import corpora, models, similarities
from collections import Counter

# list up all the reviews in NV
#f = open("review_phoenix_trunc.json")
f = open("review_3466.json")
reviews = json.load(f)
f.close()

# extract review text
theta = {}
documents = []

# remove stop words and tokenize
stoplist = set('and or but as because since then while here there when what who whose where whom which why how a an the for of to out at about on over in into onto by from up down until till before after back forth front between with away off around above below beneath within without than am is isn\'t be are aren\'t was wasn\'t were have haven\'t has hasn\'t had hadn\'t been being that this these those they i you we he she it their my your our his her its them me us him mine yours ours hers theirs i\'m you\'re it\'s there\'s here\s i\'ve i\'d i\'ll do don\'t did didn\'t does doesn\'t will won\'t would wouldn\'t shall should shouldn\'t can cannot can\'t could couldn\'t if therefore thus however although though not all no both each either none every some any again other others everyone someone another few little just also too so such very much only quite more lot once twice times many most long longer short shorter whole entire even actually still sure really especially rather now later always usually never ever sometimes often rarely good better best great greatest bad nice nicer nicest well worse worst today tomorrow yesterday thing things something nothing everything go goes went gone going come comes came coming take takes took taken taking get gets got gotten getting want wants wanted wanting make makes made making know knows knew known knowing use uses used using think thinks thought thinking read reads reading buy buys bought buying say says said saying look looks looked looking like likes liked feel feels felt feeling give gives gave given giving ask asks asked asking state states stated stating tell tells told telling let lets letting find finds found finding call calls called calling see sees saw seen seeing put puts putting seem seems seemd try tries tried trying keep keeps kept keeping speak speaks spoke spoken speaking recommend recommends recommended recommending show shows showed showing move moves moved moving order orders ordered ordering love loves loved loving guess guesses guessed guessing right left pretty set hot cold own favorite one two three four five six seven eight nine ten last next previous first second third fourth fivth sixth seventh eighth ninth tenth 1 2 3 4 5 6 7 8 9 0 - -- & ! @ # + . $ $10 :) : dr. per bit n way year years month months day days hour hours week weeks minute minutes second seconds guy kind enough far close furthher farther near full half definitely ok side yes no closed miss w/ that\'s they\'re boyfriend restaurant'.split())

#texts = [[word.rstrip('.,') for word in document.lower().split() if word.rstrip('.,') not in stoplist and word.rstrip('.,') != '']
#         for document in documents]

for review in reviews:
	#sentences = review["text"].split('.');
	#for sentence in sentences:
	if review["user_id"] not in theta:
		theta[review["user_id"]] = {}
			
	if review["business_id"] not in theta[review["user_id"]]:
		theta[review["user_id"]][review["business_id"]] = {}

	txt = review["text"];
	words = []
	for word in txt.lower().split():
		if word.rstrip('.,') not in stoplist and word.rstrip('.,') != '':
			words.append(word.rstrip('.,'))

	theta[review["user_id"]][review["business_id"]]["review"] = words;


# keep the top N frequent words and remove the other minors
#all_tokens = sum(texts, [])
#common_tokens = []
#counter = Counter(all_tokens)
#for word, cnt in counter.most_common():
#	common_tokens.append(word)

#texts = [[word for word in text if word in common_tokens]
#         for text in texts]

# if the text becomes empty, remove it.
#texts = [text for text in texts if len(text) > 0]

# finally, we get dictionary.
#dictionary = corpora.Dictionary(texts)

# Since creating a dictionary is very time consuming, wa save the dictionary so that we can reuse it later.
#dictionary.save("review.dict")

# create corpus
#corpus = [dictionary.doc2bow(text) for text in texts]

# Since creating a corpus is also very time consuming, we save the corpora.
#corpora.MmCorpus.serialize('review.mm', corpus)

###############################################################
# the following part can be done separately

# load dictionary
dictionary = corpora.Dictionary.load("review.dict")

# load corpus
corpus = corpora.MmCorpus("review.mm")

# run online Latent Direchlet Allocation
#numTopics = 20;

for numTopics in range(20, 21):
	topicFN = 'lda_topics_' + str(numTopics) + '.txt'
	resultFN = 'lda_results_' + str(numTopics) + '.txt'
	lda = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics = numTopics, update_every=1, passes=50)
	topics = lda.show_topics(numTopics)

	f = open(topicFN, 'w')
	for topic in topics:
		f.write(topic + "\n")
	f.close()

	f = open(resultFN, 'w')
	for user_id, busi_arr in theta.items():
		for business_id, review_arr in busi_arr.items():
			words = review_arr["review"];
			doc_lda = lda[dictionary.doc2bow(words)];
			theta[user_id][business_id]["result"] = doc_lda;

	#pprint.pprint(theta);
	json.dump(theta, f);


	# write the result to a file
	#for topic in topics:
		#f.write(topic + "\n\n")
	f.close()



