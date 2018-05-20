from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections 
import math

import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import numpy as np

from six.moves import urllib
import tensorflow as tf

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
	'--log_dir',
	type=str,
	default=os.path.join(current_path, 'log'),
	help='The log dir for tensorboard'
)

FLAGS, unparsed = parser.parse_known_args()
print(FLAGS, unparsed)

if not os.path.exists(FLAGS.log_dir):
	os.makedirs(FLAGS.log_dir)
	
url = "http://mattmahoney.net/dc/"

def maybe_download(filename, expected_bytes):
    #local_filename = os.path.join(gettempdir(), filename)
    local_filename = os.path.join("data/", filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
    statsinfo = os.stat(local_filename)	
    if statsinfo.st_size == expected_bytes:
        print("Found and verified", filename)
    else:
        print(statsinfo.st_size)
        raise Exception('Failed to verify ' + local_filename)

    return local_filename

filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print(vocabulary[:10])
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words, n_words):
    # For set usage
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words-1)) 
    # dictionary is a word-idx mapping
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data  = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary
print("Most common words (+UNK)", count[:5])
print("Sample data", data[:10], [reversed_dictionary[i] for i in data[:10]])
        
data_index = 0
# Step3: Generate a training batch for the skip-gram model
def generate_batch(batch_size, num_skips, skip_window):
    global  data_index
    assert num_skips <= 2 * skip_window
    assert batch_size % num_skips == 0

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1 
    buffer = collections.deque(maxlen=span) 
    if data_index + span > len(data):
        data_index = 0
    
    buffer.extend(data[data_index: data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_word = [ w for w in range(span) if w != skip_window] # omit the center word
        words_to_use = random.sample(context_word, num_skips)

        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]

        if data_index == len(data):
            buffer.extend(data[0: span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size = 8, num_skips = 2, skip_window = 1)
for i in range(8):
    print(batch[i], reversed_dictionary[batch[i]], '->', labels[i, 0], reversed_dictionary[labels[i, 0]])

    
# Step 4: Build and train a skip-gram model
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64 # num of negative examples to sample

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph()

with graph.as_default():
    # input data
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    with tf.device('/gpu:0'):
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))

        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights = nce_weights, 
                    biases = nce_biases,
                    labels = train_labels, 
                    inputs = embed,
                    num_sampled = num_sampled,
                    num_classes = vocabulary_size))
    tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    
    # Merge all summaries
    merged = tf.summary.merge_all()

    # Add variable initializer
    init = tf.global_variables_initializer()
    
    # Create a saver
    saver = tf.train.Saver()

# Step 5: Begin training
num_steps = 10001

with tf.Session(graph=graph) as session:
    # Open a writer to write summarise
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph )
    print('Initialized')
    
    # Initialize all variables before using them
    init.run() 

    averge_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        run_metadata = tf.RunMetadata()
        _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict = feed_dict, run_metadata = run_metadata)

        averge_loss += loss_val
        writer.add_summary(summary, step) 
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)
        
        if step % 2000 == 0:
            if step > 0:
                averge_loss /= 2000
            print('Average loss at step ', step, ': ', averge_loss)
            averge_loss = 0

        
        # Eval sim
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k+1]
                log_str = 'Nearest to %s: ' % valid_word
                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word) 
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    
    # Write the corresponding labels for the embeddings
    with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
        for i in xrange(vocabulary_size):
            f.write(reversed_dictionary[i] + "\n")
    
    saver.save(session, os.path.join(FLAGS.log_dir, 'model,ckpt'))
    
