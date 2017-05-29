class vocab(object):

	def __init__(self):
		self.vocabulary = {}
		self.count = 0
		self.max_len = 0

	def readVocab(self, filePath):
		f = open(filePath, 'r')
		line = f.readline()
		while line:
			sentence = line.strip()
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
vocabObj.readVocab("../data/train/source.txt")
vocabObj.readVocab("../data/validation/source.txt")
vocabObj.saveVocab("../data/source_vocab.txt")
print(vocabObj.max_len)
vocabObj1 = vocab()
vocabObj1.readVocab("../data/train/target.txt")
vocabObj1.readVocab("../data/validation/target.txt")
vocabObj1.saveVocab("../data/target_vocab.txt")
print(vocabObj1.max_len)
