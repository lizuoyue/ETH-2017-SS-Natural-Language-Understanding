#!/usr/bin/env python3
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains a seq2seq model.

WORK IN PROGRESS.

Implement "Abstractive Text Summarization using Sequence-to-sequence RNNS and
Beyond."

"""
import numpy as np
import sys
import time
import struct
import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model
from tensorflow.core.example import example_pb2


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           '', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('article_key', 'article',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'headline',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory for eval.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences', 2,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_integer('beam_size', 4,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', False,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
  return running_avg_loss


def _Train(model, data_batcher):
  """Runs model training."""
  with tf.device('/cpu:0'):
    model.build_graph()
    saver = tf.train.Saver()
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while not sv.should_stop() and step < FLAGS.max_run_steps:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
       loss_weights, _, _) = data_batcher.NextBatch()
      """for i in article_batch[0]:
        if (i!=1):
          print(vocab.IdToWord(i)),
      print (" ")"""
      (_, summaries, loss, train_step) = model.run_train_step(
          sess, article_batch, abstract_batch, targets, article_lens,
          abstract_lens, loss_weights)

      summary_writer.add_summary(summaries, train_step)
      running_avg_loss = _RunningAvgLoss(
          running_avg_loss, loss, summary_writer, train_step)
      step += 1
      if step % 100 == 0:
        summary_writer.flush()
    sv.Stop()
    return running_avg_loss

def sentence_to_id(sen,vocab,limit,pad):
  t = sen.split(" ")
  ans = []
  for i in t:
    ans.append(vocab.WordToId(i))
    if (len(ans)==limit):
      break
  for i in range(len(t),limit):
    ans.append(pad);
  target = ans[1:]
  target.append(pad)
  #print ("target")
  #print (target)
  return [np.array(ans),min(len(t),limit),np.array(target)];



def _Eval(model, data_batcher, vocab=None,hps=None):
  
  """
  reader = open(FLAGS.data_path, 'rb')
  t=0
  while t<5:
    t=t+1
    len_bytes = reader.read(8)
    if not len_bytes:
      return
    str_len = struct.unpack('q', len_bytes)[0]
    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    tf_example = example_pb2.Example.FromString(tf_example_str)
    
    article = (tf_example.features.feature["article"].bytes_list.value[0])
    [article,article_lens,_] = sentence_to_id(article,vocab,hps.enc_timesteps,1)
    article_batch = np.array([article])

    abstract = (tf_example.features.feature["abstract"].bytes_list.value[0])
    [abstract,abstract_lens,target] = sentence_to_id(abstract,vocab,hps.dec_timesteps,2)
    abstract_batch = np.array([abstract])
    targets = np.array([target])


    loss_weights = np.zeros(
        (hps.batch_size, hps.dec_timesteps), dtype=np.float32)
    for i in range(0,abstract_lens):
      loss_weights[0][i]=1;
    abstract_lens = np.array([abstract_lens])
    article_lens = np.array([article_lens])
    print (abstract_batch.shape)
    print (abstract_lens.shape)

  
  reader.close()

  
  
  
  
  t=0
  while t<5:
    t+=1
    (article_batch, abstract_batch, targets, article_lens, abstract_lens,
     loss_weights, _, _) = data_batcher.NextBatch()
    print (article_batch.shape)
    print (article_lens.shape)
    print (abstract_batch.shape)
    print (targets.shape)

    for i in abstract_batch[0]:
      if (i!=2):
        print(vocab.IdToWord(i)),
    print (" ")
  a=a
  """
  
  



  reader = open(FLAGS.data_path, 'rb')
  out = open("perplexity-result.txt","w")
  out2 = open("senten-result.txt","w")
  t=0
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  step = 0
  while True:
    #time.sleep(FLAGS.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)



    t=t+1
    len_bytes = reader.read(8)
    if not len_bytes:
      return
    str_len = struct.unpack('q', len_bytes)[0]
    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    tf_example = example_pb2.Example.FromString(tf_example_str)
    
    article = (tf_example.features.feature["article"].bytes_list.value[0])
    [article,article_lens,_] = sentence_to_id(article,vocab,hps.enc_timesteps,1)
    article_batch = np.array([article])

    abstract = (tf_example.features.feature["abstract"].bytes_list.value[0])
    [abstract,abstract_lens,target] = sentence_to_id(abstract,vocab,hps.dec_timesteps,2)
    abstract_batch = np.array([abstract])
    targets = np.array([target])


    loss_weights = np.zeros(
        (hps.batch_size, hps.dec_timesteps), dtype=np.float32)
    for i in range(0,abstract_lens):
      loss_weights[0][i]=1;
    abstract_lens = np.array([abstract_lens])
    article_lens = np.array([article_lens])
    """
    print (abstract_batch)
    print (abstract_lens)
    print (article_batch)
    print (article_lens)
    print (loss_weights)


    (article_batch2, abstract_batch2, targets2, article_lens2, abstract_lens2,
     loss_weights2, _, _) = data_batcher.NextBatch()
    print (abstract_batch2)
    print (abstract_lens2)
    print (article_batch2)
    print (article_lens2)
    print (loss_weights2)
    """


    for i in article_batch[0]:
      if (i!=1):
        out2.write(vocab.IdToWord(i)+" "),
    out2.write("\t")
    for i in abstract_batch[0]:
      if (i!=2):
        out2.write(vocab.IdToWord(i)+" "),
    out2.write("\n")

    (summaries, loss, train_step) = model.run_eval_step(
        sess, article_batch, abstract_batch, targets, article_lens,
        abstract_lens, loss_weights)
    sys.stdout.write('loss: %f\n' % loss)
    out.write('%f\n' % (2**loss))
    tf.logging.info(
        'article:  %s',
        ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
    tf.logging.info(
        'abstract: %s',
        ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))

    summary_writer.add_summary(summaries, train_step)
    running_avg_loss = _RunningAvgLoss(
        running_avg_loss, loss, summary_writer, train_step)
    if step % 100 == 0:
      summary_writer.flush()

  reader.close()
  out.close()
  out2.close()


def main(unused_argv):
  vocab = data.Vocab(FLAGS.vocab_path, 1000000)
  # Check for presence of required special tokens.
  assert vocab.CheckVocab(data.PAD_TOKEN) > 0
  assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
  assert vocab.CheckVocab(data.SENTENCE_START) > 0
  assert vocab.CheckVocab(data.SENTENCE_END) > 0

  batch_size = 4
  if FLAGS.mode == 'decode':
    batch_size = FLAGS.beam_size
  if FLAGS.mode == 'eval':
    batch_size = 1

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=0.15,  # learning rate
      batch_size=batch_size,
      enc_layers=4,
      enc_timesteps=100,
      dec_timesteps=100,
      min_input_len=2,  # discard articles/summaries < than this
      num_hidden=256,  # for rnn cell
      emb_dim=128,  # If 0, don't use embedding
      max_grad_norm=2,
      num_softmax_samples=4096)  # If 0, no sampled softmax.

  batcher = batch_reader.Batcher(
      FLAGS.data_path, vocab, hps, FLAGS.article_key,
      FLAGS.abstract_key, FLAGS.max_article_sentences,
      FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
      truncate_input=FLAGS.truncate_input)
  tf.set_random_seed(FLAGS.random_seed)

  if hps.mode == 'train':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Train(model, batcher)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Eval(model, batcher, vocab=vocab,hps =hps)
  elif hps.mode == 'decode':
    decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
    decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
    decoder.DecodeLoop()


if __name__ == '__main__':
  tf.app.run()
