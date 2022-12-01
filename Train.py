#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import pickle, os
import random
import Config

import tensorflow_addons as tfa

# from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example

config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True

total_step = 248 #get value from output of Preprocess.py file

config = Config.Config()

word_vec = pickle.load(open('word_vec.pkl', 'rb'))
vocab = pickle.load(open('word_voc.pkl','rb'))

config.vocab_size = len(vocab)

class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.size = size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.lr = config.learning_rate

        def lstm_cell(): 
            LSTM_cell = tf.keras.layers.LSTMCell(size, dropout = config.keep_prob if is_training else 1.0)
            return LSTM_cell
        
        self.cells = tf.keras.layers.StackedRNNCells([lstm_cell() for _ in range(config.num_layers)])

        self.embedding = word_vec # pickle.load(open('word_vec.pkl', 'rb'))
        # inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        # keyword_inputs = tf.nn.embedding_lookup(self.embedding, self._input_word)

        # with tf.device("/cpu:0"):
        #     embedding = tf.compat.v1.get_variable('word_embedding', [vocab_size, config.word_embedding_size], trainable=True, initializer=tf.compat.v1.constant_initializer(word_vec), use_resource=False)
        #     inputs = tf.nn.embedding_lookup(params=embedding, ids=self._input_data)#返回一个tensor，shape是(batch_size, num_steps, size)
        #     keyword_inputs = tf.nn.embedding_lookup(params=embedding, ids=self._input_word)

        # TODO: add functionality later
        # if is_training and config.keep_prob < 1:
        #     inputs = tf.nn.dropout(inputs, rate=1 - (config.keep_prob))

        self.gate = tf.ones([self.batch_size, config.num_keywords])
        self.atten_sum = tf.zeros([self.batch_size, config.num_keywords])
        self.u_f = tf.Variable(tf.random_uniform_initializer(shape=[config.num_keywords * config.word_embedding_size, config.num_keywords], dtype=tf.float32))
        # self.u_f = tf.compat.v1.get_variable("u_f", [config.num_keywords*config.word_embedding_size, config.num_keywords], use_resource=False)

        self.u = tf.Variable(tf.random_uniform_initializer(shape=[self.size, 1], dtype=tf.float32))
        self.w1 = tf.Variable(tf.random_uniform_initializer(shape=[self.size, self.size], dtype=tf.float32))
        self.w2 = tf.Variable(tf.random_uniform_initializer(shape=[config.word_embedding_size, self.size], dtype=tf.float32))
        self.b1 = tf.Variable(tf.random_uniform_initializer(shape=[self.size], dtype=tf.float32))

        self.softmax_w = tf.Variable(tf.random_uniform_initializer(shape=[self.size, self.vocab_size], dtype=tf.float32))
        self.softmax_b = tf.Variable(tf.random_uniform_initializer(shape=[self.vocab_size], dtype=tf.float32))

    def forward(self, input_data, init_output, mask, keyword_inputs, is_training=False):
        # probably MTA part
        res1 = tf.sigmoid(tf.matmul(tf.reshape(keyword_inputs, [self.batch_size, -1]), self.u_f))
        phi_res = tf.reduce_sum(input_tensor=mask, axis=1, keepdims=True) * res1
            
        self.output1 = phi_res
            
        outputs = []
        output_state = init_output.copy()
        # state = self._initial_state
        entropy_cost = []

        for time_step in range(self.num_steps):
            vs = []
            for s2 in range(config.num_keywords):
                vi = tf.matmul(tf.tanh(tf.add(tf.add(
                    tf.matmul(output_state, self.w1),
                    tf.matmul(keyword_inputs[:, s2, :], self.w2)), self.b)), self.u)
                vs.append(vi * self.gate[:, s2:s2+1])
                
                attention_vs = tf.concat(vs, axis=1)
                prob_p = tf.nn.softmax(attention_vs)
                
                self.gate = self.gate - (prob_p / phi_res)
                
                atten_sum += prob_p * mask[:, time_step:time_step+1]
                
                mt = tf.add_n([prob_p[:, i:i+1]*keyword_inputs[:, i, :] for i in range(config.num_keywords)])

                # if time_step > 0: tf.compat.v1.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cells(tf.concat([input_data[:, time_step, :], mt], axis=1), state) 
                outputs.append(cell_output)
                output_state = cell_output
            
            end_output = cell_output
            
        self.output2 = atten_sum    
        output = tf.reshape(tf.concat(outputs, axis=1), [-1, self.size])
        print("OUTPUT", output.shape)
        print("SOFTMAX W", self.softmax_w.shape)
        print("SOFTMAX B", self.softmax_b.shape)
        logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        _ , logits_dim = logits.shape
        logits = tf.reshape(logits, [self.batch_size, -1, logits_dim])
        print("LOGITS", logits.shape)
        print("TARGETS", self._targets.shape)
        print("MASK", self._mask.shape)

        if not is_training:
            prob = tf.nn.softmax(logits)
            self._sample = tf.argmax(input=prob, axis=1)
            return
        return logits, phi_res, atten_sum

    def loss_calc(self, logits, targets, mask, phi_res, atten_sum):
        # Loss calculation
        loss = tfa.seq2seq.sequence_loss(
            logits,
            targets,
            mask, average_across_timesteps=False)
        
        cost1 = tf.reduce_sum(input_tensor=loss)
        cost2 = tf.reduce_sum(input_tensor=(phi_res - atten_sum)**2)

        return (cost1 + 0.1 * cost2) / self.batch_size
        
    
def main():
    batch_size = config.batch_size
    num_steps = config.num_steps
    
    def decode_fn(record_bytes):
        return tf.io.parse_single_example(
            # Data
            record_bytes,

            # Schema
            {
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'input_data': tf.io.FixedLenSequenceFeature([batch_size*num_steps], tf.int64, allow_missing=True, default_value=0),
                'target': tf.io.FixedLenSequenceFeature([batch_size*num_steps], tf.int64, allow_missing=True, default_value=0),
                'mask': tf.io.FixedLenSequenceFeature([batch_size*num_steps], tf.float32, allow_missing=True, default_value=0.0),
                'key_words': tf.io.FixedLenSequenceFeature([batch_size*config.num_keywords], tf.int64, allow_missing=True, default_value=0)
            }
        )
    train_dataset = tf.data.TFRecordDataset(["coverage_data"]).shuffle(batch_size).repeat(None)

    model = Model(is_training=True, config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        init_output = np.zeros((model.batch_size, model.size))
        # Iterate over the batches of the dataset.
        for step, (input_data, target, mask, key_words) in enumerate(train_dataset.map(decode_fn)):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                # logits = model(x_batch_train, training=True)  # Logits for this minibatch
                logits, phi_res, atten_sum = model.forward(input_data, init_output, mask, key_words, is_training=False)

                # Compute the loss value for this minibatch.
                loss_value = model.loss_calc(logits, target, mask, phi_res, atten_sum)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
            
if __name__ == "__main__":
    main()