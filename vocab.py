class vocab(object):

	def __init__(self):
		self.vocabulary = {}
		self.count = 0
		self.max_len = 0

	def readVocab(self, filePath):
		f = open(filePath, 'r')
		line = f.readline()
		while line:
			sentences = line.strip().split('\t')
			for sentence in sentences:
				self.count += 1
				words = sentence.split(' ')
				self.max_len = max(self.max_len, len(words))
				for word in words:
					if word not in self.vocabulary:
						self.vocabulary[word] = 0
					self.vocabulary[word] += 1
			line = f.readline()
		self.vocabulary["<eos>"] = self.count
		f.close()

	def saveVocab(self, fileName):
		vocabList = sorted(self.vocabulary.items(), key = lambda d: d[1], reverse = True)
		out = open(fileName, 'w')
		for i in range(len(vocabList)):
			out.write(vocabList[i][0] + "\n")
		out.close()

vocabObj = vocab()
vocabObj.readVocab("../data/Training_Shuffled_Dataset.txt")
vocabObj.readVocab("../data/Validation_Shuffled_Dataset.txt")
vocabObj.saveVocab("vocab.txt")
print(vocabObj.max_len)
