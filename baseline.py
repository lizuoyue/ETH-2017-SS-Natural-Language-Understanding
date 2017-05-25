from __future__ import print_function

import tensorflow as tf
import numpy as np

# Deactivate the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Learning parameters
learning_rate = 1e-2
training_iter = 1e+6
batch_size    = 64
display_step  = 10

# Network parameters
max_seq_len   = 50
state_size    = 512
vocab_size    = 58000
emb_size      = 100

# Construct vocabulary index dictionary
vocab = {}
look_up = []
f = open("vocab.txt", 'r')
line = f.readline()
idx = 0
while line:
	look_up.append(line.strip())
	vocab[look_up[idx]] = idx;
	idx += 1
	line = f.readline()
f.close()

# Graph input and output
x = tf.placeholder(tf.int32, [batch_size, max_seq_len])
y = tf.placeholder(tf.int32, [batch_size, max_seq_len])

# Sequence length and prediction
len_i = tf.placeholder(tf.int32, [batch_size]) # include <eos>
len_o = tf.placeholder(tf.int32, [batch_size]) # exclude <eos>

# Embedding weights, output weights and biases
emb = tf.Variable(tf.random_normal([vocab_size, emb_size]))
w = tf.Variable(tf.random_normal([state_size, vocab_size]))
b = tf.Variable(tf.random_normal([vocab_size]))

# LSTM cells
with tf.variable_scope("lstm_i"):
	cell_i = tf.contrib.rnn.BasicLSTMCell(state_size)
with tf.variable_scope("lstm_o"):
	cell_o = tf.contrib.rnn.BasicLSTMCell(state_size)

# First LSTM
emb_i = tf.nn.embedding_lookup(emb, x)
input_i = tf.unstack(emb_i, axis = 1)
with tf.variable_scope("lstm_i"):
	output, state_i = tf.contrib.rnn.static_rnn(cell_i, input_i, dtype = tf.float32, sequence_length = len_i)
output = tf.stack(output)
output = tf.transpose(output, [1, 0, 2])
output = tf.reshape(output, [-1, state_size])
index = tf.range(0, batch_size) * max_seq_len + (len_i - 1)
output = tf.gather(output, index)
encode = tf.matmul(output, w) + b

# Second LSTM
emb_o = tf.nn.embedding_lookup(emb, y)
input_o = tf.unstack(emb_o, axis = 1)
with tf.variable_scope("lstm_o"):
	output, state_o = tf.contrib.rnn.static_rnn(cell_o, input_o, dtype = tf.float32, sequence_length = len_o, initial_state = state_i)
output = tf.stack(output)
output = tf.transpose(output, [1, 0, 2])
output = tf.reshape(output, [-1, state_size])
true_y = y[:, 0]
for i in range(batch_size):
	if i == 0:
		index = i * max_seq_len + tf.range(len_o[i])
	else: 
		index = tf.concat([index, i * max_seq_len + tf.range(len_o[i])], axis = 0)
	true_y = tf.concat([true_y, y[i, 1: len_o[i] + 1]], axis = 0)
output = tf.gather(output, index)
output = tf.stack(output)
pred_logits = tf.matmul(output, w) + b
total_logits = tf.concat([encode, pred_logits], axis = 0)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=total_logits, labels=tf.reshape(true_y, [-1])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

f = open("../data/Training_Shuffled_Dataset.txt", 'r')
# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step * batch_size < training_iter:
		batch_x = []
		batch_y = []
		len_x = []
		len_y = []
		while len(batch_x) < batch_size:

			line = f.readline()
			if not line:
				f.close()
				f = open("../data/Training_Shuffled_Dataset.txt", 'r')
				line = f.readline()

			sentence = line.strip().split('\t')
			word = [None, None, None]
			word[0] = sentence[0].split(' ')
			word[1] = sentence[1].split(' ')
			word[2] = sentence[2].split(' ')
			if (len(word[0]) < max_seq_len - 1 and len(word[1]) < max_seq_len - 1 and len(word[2]) < max_seq_len - 1):
				code = [[], [], []]
				code_r = [[], [], []]
				length = [0, 0, 0]
				for i in range(3):
					for item in word[i]:
						code[i].append(vocab[item])
						code_r[i].append(vocab[item])
					code_r[i].reverse()
					code[i].append(vocab["<eos>"])
					code_r[i].append(vocab["<eos>"])
					length[i] = len(code[i])
					while len(code[i]) < max_seq_len:
						code[i].append(-1+vocab_size)
						code_r[i].append(-1+vocab_size)
				
				batch_x.append(code_r[0])
				batch_x.append(code_r[1])
				len_x.append(length[0])
				len_x.append(length[1])

				batch_y.append(code[1])
				batch_y.append(code[2])
				len_y.append(length[1] - 1)
				len_y.append(length[2] - 1)

		batch_x = np.array(batch_x)
		batch_y = np.array(batch_y)
		len_x = np.array(len_x)
		len_y = np.array(len_y)

		feed_dict={x: batch_x, y: batch_y, len_i: len_x, len_o: len_y}
		sess.run(optimizer, feed_dict=feed_dict)
		if step % display_step == 0:
			loss = sess.run(cost, feed_dict=feed_dict)
			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
				"{:.6f}".format(loss))
		step += 1
	print("Optimization Finished!")
