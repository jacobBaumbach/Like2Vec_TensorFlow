from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tsne import tsne
import matplotlib.pyplot as plt


class Like2Vec(object):
    def __init__(self,user_item,name_index_dict,wpv,wl,add_coeff = .01,axis = 0):
        """
        Initialize an instance of Like2Vec
        
        INPUT:
            user_item : user(rows) item(columns) matrix, 2 dimensional numpy array 
            name_index_dict : label (key) index (value), dictonary(string:int)
            wpv : random walks per value, int
            wl : length of random walk, int
            add_coeff : laplace smoothing coefficient (amount you will add to each numerator of proportion matrix), double
            axis : when 0 you embeddings will be generated for users, else embeddings will be generated for items
        """
        self.user_item = user_item
        self.axis_matrix = self.user_item.dot(self.user_item.T) if axis==0 else self.user_item.T.dot(self.user_item)
        self.final_matrix = self.laplace_smoothing(add_coeff)
        self.name_index_dict = name_index_dict
        self.index_name_dict = dict(zip(self.name_index_dict.values(),self.name_index_dict.keys()))
        self.labels = [i.decode('ascii',errors="ignore") for i in self.name_index_dict.keys()]
        self.data = self.rando_walks(wpv,wl)
        self.data_index = 0
        self.graph = tf.Graph()
        
    def laplace_smoothing(self,add_coeff):
        """
        Laplace Smoothing is performed on the proportion matrix used to generate random walks
        
        INPUT:
            add_coeff : the coefficient that will be added to the numerator so for each given user or item
                        all the other users or items will have nonzero proportions, double
        OUTPUT:
            2 dimensional array of size either user x user or item x item containing the laplace smoothed proportions
        """
        return (self.axis_matrix+add_coeff)/(np.sum(self.axis_matrix,axis=1)+add_coeff*len(self.axis_matrix))
    
    def rando_walks(self,wpv,wl):
        """
        For each either user or item, wpv random walks will be generated for each user and item and those random walks
        will all be of length wl.  These random walks are used to generate the embeddings.
        
        INPUT:
            wpv : random walks per value, int
            wl : length of random walk, int
            
        OUTPUT:
            list of lists of lists, the outmost list contains a list for every user or item and that list contains
            wpv lists where each of those lists contains wl integers of the either users or items visited in the
            random walk
        """
        rng = xrange(len(self.final_matrix))
        wpvrng = xrange(wpv)
        wlrng = xrange(wl)
        def rw(idx):
            def repeatRw():
                rwLst = [idx]
                def w(nuIdx):
                    rwLst.append(np.random.choice(rng,p=self.final_matrix[nuIdx]/np.sum(self.final_matrix[nuIdx])))
                map(w,wlrng)
                return rwLst
            return [repeatRw() for _ in wpvrng]
        return list(map(rw,rng))

    def _generate_batch(self,batch_size, num_skips, skip_window):
        """
        Creates the minibatch that embeddings will be trained on for a given iteration
        
        INPUT:
            batch_size : the number of samples the embeddings will be trained on for a given iteration
            num_skips : How many times to reuse an input to generate a label.
            skip_window : How many words to consider left and right
            
        OUTPUT:
            batch : 1 dimensional array of length batch_size that will be the input to train the embeddings
            labels : 1 dimensional array of length batch_size that will be the output the embeddings try to match
                     during training
        """
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(random.choice(self.data[self.data_index]))#CHANGE!
            self.data_index = (self.data_index + 1) % len(self.final_matrix)

        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)

                batch[i * num_skips + j] = buffer[target][skip_window]
                labels[i * num_skips + j, 0] = buffer[target][0]
            buffer.append(random.choice(self.data[self.data_index]))
            self.data_index = (self.data_index + 1) % len(self.final_matrix)
        return batch, labels

    def _build_skip_gram(self,batch_size = 128,embedding_size = 128,learn_rate=1.0,num_sampled = 64,num_skips = 2,
                         skip_window = 1,valid_size = 16 ,valid_window = 100 ):
        """
        Creates the inputs needed to begin training the embeddings
        
        INPUT:
            batch_size : the number of samples the embeddings will be trained on for a given iteration, int
            embedding_size : Dimension of the embedding vector, int
            learn_rate : the learning rate used during optimization, double
            num_sampled : Number of negative examples to sample, int
            num_skips : How many times to reuse an input to generate a label, int
            skip_window : How many words to consider left and right, int
            valid_size : Random set of users or items to evaluate similarity on, int
            valid_window : Only pick dev samples in the head of the distribution, int
            
        OUTPUT:
            loss : Compute the average NCE loss for the batch. 
                   tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
            normalized_embeddings : tensorflow object that will hold the value of either the user or item embeddings.
            optimizer : Construct the SGD optimizer using a learning rate of learn_rate
            similarity : Computes the cosine similarity between minibatch examples and all embeddings
            train_inputs : placeholder for input data used to train embeddings
            train_labels : placeholder for output the embeddings try to match during training
            valid_examples : randomly chooses valid_size out of the numbers between 0 and valid_window to 
        """
         
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)

        with self.graph.as_default():
            vocabulary_size = len(self.final_matrix)
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
              embeddings = tf.Variable(
                  tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
              embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
              nce_weights = tf.Variable(
                  tf.truncated_normal([vocabulary_size, embedding_size],
                                      stddev=1.0 / math.sqrt(embedding_size)))
              nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                               num_sampled, vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)
        return loss,normalized_embeddings,optimizer,similarity,train_inputs,train_labels,valid_examples
        
    def fit(self,batch_size = 128,embedding_size = 128,learn_rate=1.0,num_sampled = 64,num_steps = 100001,
                         num_skips = 2,skip_window = 1,valid_size = 16 ,valid_window = 100,
                         verbose=False):
        """
        Generates embeddings
        
        INPUT:
            batch_size : the number of samples the embeddings will be trained on for a given iteration, int
            embedding_size : Dimension of the embedding vector, int
            learn_rate : the learning rate used during optimization, double
            num_sampled : Number of negative examples to sample, int
            num_steps : number of iterations to train the embeddings, int
            num_skips : How many times to reuse an input to generate a label, int
            skip_window : How many words to consider left and right, int
            valid_size : Random set of users or items to evaluate similarity on, int
            valid_window : Only pick dev samples in the head of the distribution, int
            verboose : True will print progress, else progress will not be printed, boolean
            
        OUTPUT:
            final_embeddings : either user x embedding_size or item x embedding_size 2 dimensional numpy array
                               containing the final embeddings for all users or items
        """
                
        loss,normalized_embeddings,optimizer,similarity,train_inputs,train_labels,valid_examples = self._build_skip_gram(batch_size,embedding_size,learn_rate,num_sampled,num_skips,
                         skip_window,valid_size,valid_window)
        
        with tf.Session(graph=self.graph) as session:
            # We must initialize all variables before we use them.
            tf.initialize_all_variables().run()

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = self._generate_batch(
                    batch_size, num_skips, skip_window)
                feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                if verbose:
                    average_loss += loss_val
                    if step % 2000 == 0:
                        if step > 0:
                            average_loss /= 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print("Average loss at step ", step, ": ", average_loss)
                        average_loss = 0
                    if step % 10000 == 0:# Note that this is expensive (~20% slowdown if computed every 500 steps)
                        sim = similarity.eval()
                        for i in xrange(valid_size):
                            valid_word = self.index_name_dict[valid_examples[i]]
                            top_k = 8 # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k+1]
                            log_str = "Nearest to %s:" % valid_word
                            for k in xrange(top_k):
                                close_word = self.index_name_dict[nearest[k]]
                                log_str = "%s %s," % (log_str, close_word)
                            print(log_str)
                            print("")
            final_embeddings = normalized_embeddings.eval()
        self.final_embeddings = final_embeddings
        return self.final_embeddings
    
    def plot_with_labels(self,plot_only = 100, title="Like2Vec meets TensorFlow", filename='tsne.png',
                         num_tsne_dims = 2, perplexity = 5.0,verbose=False):
        """
        randomly chooses some of the users or items and plots them using tsne
        
        INPUT:
            plot_only : the number of users or items you would like plotted,int
            title : the title of the plot generated, str
            filename : the name you would like the file saved under, str
            num_tsne_dims : number of dimensions, int
            perplexity : the perplexity used in generating tsne, recommended to be between 5.0-50.0, double
            verbose : whether or not to print the progress of tsne
        
        OUTPUT:
            Your plot will be saved under the name and in the location you passed in filename
        """
        selected_rows = np.sort(np.random.choice(range(self.final_embeddings.shape[0]),plot_only,replace=False))
        labels = [self.labels[i] for i in selected_rows]
        low_dim_embs = tsne(self.final_embeddings[selected_rows], num_tsne_dims,
                            self.final_embeddings.shape[1], perplexity,verbose)
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  #in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i,:]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.title(title)
        plt.savefig(filename)