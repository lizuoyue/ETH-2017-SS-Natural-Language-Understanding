f = open("../data/Training_Shuffled_Dataset.txt", 'r')
vocabulary = {}

line = f.readline()
while line:

	sentences = line.strip().split('\t')
	for sentence in sentences:
		words = sentence.split(' ')
		for word in words:
			if word not in vocabulary:
				vocabulary[word] = 0;
			vocabulary[word] += 1;
	line = f.readline()

f.close()

vocabulary = sorted(vocabulary.items(), key = lambda d: d[1], reverse = True)
print(len(vocabulary))
