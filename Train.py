#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import pickle5 as pickle
import os
import random
import Config

from tqdm import tqdm

# import tensorflow_addons as tfa

# from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example

config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True

total_step = 248 #get value from output of Preprocess.py file

config = Config.Config()

word_vec = pickle.load(open('word_vec.pkl', 'rb'))
vocab = pickle.load(open('word_voc.pkl','rb'))

config.vocab_size = len(vocab)

def sequence_loss(
    logits,
    targets,
    weights,
    average_across_timesteps: bool = True,
    average_across_batch: bool = True,
    sum_over_timesteps: bool = False,
    sum_over_batch: bool = False,
    softmax_loss_function = None,
    name = None,
) -> tf.Tensor:
    if len(logits.shape) != 3:
        raise ValueError(
            "Logits must be a [batch_size x sequence_length x logits] tensor"
        )

    targets_rank = len(targets.shape)
    if targets_rank != 2 and targets_rank != 3:
        raise ValueError(
            "Targets must be either a [batch_size x sequence_length] tensor "
            + "where each element contains the labels' index"
            + "or a [batch_size x sequence_length x num_classes] tensor "
            + "where the third axis is a one-hot representation of the labels"
        )

    if len(weights.shape) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] tensor")

    if average_across_timesteps and sum_over_timesteps:
        raise ValueError(
            "average_across_timesteps and sum_over_timesteps cannot "
            "be set to True at same time."
        )
    if average_across_batch and sum_over_batch:
        raise ValueError(
            "average_across_batch and sum_over_batch cannot be set "
            "to True at same time."
        )
    if average_across_batch and sum_over_timesteps:
        raise ValueError(
            "average_across_batch and sum_over_timesteps cannot be set "
            "to True at same time because of ambiguous order."
        )
    if sum_over_batch and average_across_timesteps:
        raise ValueError(
            "sum_over_batch and average_across_timesteps cannot be set "
            "to True at same time because of ambiguous order."
        )
    with tf.name_scope(name or "sequence_loss"):
        num_classes = tf.shape(input=logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        if softmax_loss_function is None:
            if targets_rank == 2:
                targets = tf.reshape(targets, [-1])
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits_flat
                )
            else:
                targets = tf.reshape(targets, [-1, num_classes])
                crossent = tf.nn.softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits_flat
                )
        else:
            targets = tf.reshape(targets, [-1])
            crossent = softmax_loss_function(labels=targets, logits=logits_flat)
        crossent *= tf.reshape(weights, [-1])
        if average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(input_tensor=crossent)
            total_size = tf.reduce_sum(input_tensor=weights)
            crossent = tf.math.divide_no_nan(crossent, total_size)
        elif sum_over_timesteps and sum_over_batch:
            crossent = tf.reduce_sum(input_tensor=crossent)
            total_count = tf.cast(tf.math.count_nonzero(weights), crossent.dtype)
            crossent = tf.math.divide_no_nan(crossent, total_count)
        else:
            crossent = tf.reshape(crossent, tf.shape(input=logits)[0:2])
            if average_across_timesteps or average_across_batch:
                reduce_axis = [0] if average_across_batch else [1]
                crossent = tf.reduce_sum(input_tensor=crossent, axis=reduce_axis)
                total_size = tf.reduce_sum(input_tensor=weights, axis=reduce_axis)
                crossent = tf.math.divide_no_nan(crossent, total_size)
            elif sum_over_timesteps or sum_over_batch:
                reduce_axis = [0] if sum_over_batch else [1]
                crossent = tf.reduce_sum(input_tensor=crossent, axis=reduce_axis)
                total_count = tf.cast(
                    tf.math.count_nonzero(weights, axis=reduce_axis),
                    dtype=crossent.dtype,
                )
                crossent = tf.math.divide_no_nan(crossent, total_count)
        return crossent
class LSTMModel(object):
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

        self.embedding = tf.cast(word_vec, tf.float32) # pickle.load(open('word_vec.pkl', 'rb'))
        # inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        # keyword_inputs = tf.nn.embedding_lookup(self.embedding, self._input_word)

        # with tf.device("/cpu:0"):
        #     embedding = tf.compat.v1.get_variable('word_embedding', [vocab_size, config.word_embedding_size], trainable=True, initializer=tf.compat.v1.constant_initializer(word_vec), use_resource=False)
        #     inputs = tf.nn.embedding_lookup(params=embedding, ids=self._input_data)#返回一个tensor，shape是(batch_size, num_steps, size)
        #     keyword_inputs = tf.nn.embedding_lookup(params=embedding, ids=self._input_word)

        # TODO: add functionality later
        # if is_training and config.keep_prob < 1:
        #     inputs = tf.nn.dropout(inputs, rate=1 - (config.keep_prob))

        self.gate = tf.Variable(tf.ones([self.batch_size, config.num_keywords]))
        self.atten_sum = tf.Variable(tf.zeros([self.batch_size, config.num_keywords]))

        random_uniform_initializer = tf.random_uniform_initializer(minval=-config.init_scale, maxval=config.init_scale)
        self.u_f = tf.Variable(random_uniform_initializer(shape=[config.num_keywords * config.word_embedding_size, config.num_keywords], dtype=tf.float32))

        self.u = tf.Variable(random_uniform_initializer(shape=[self.size, 1], dtype=tf.float32))
        self.w1 = tf.Variable(random_uniform_initializer(shape=[self.size, self.size], dtype=tf.float32))
        self.w2 = tf.Variable(random_uniform_initializer(shape=[config.word_embedding_size, self.size], dtype=tf.float32))
        self.b1 = tf.Variable(random_uniform_initializer(shape=[self.size], dtype=tf.float32))

        self.softmax_w = tf.Variable(random_uniform_initializer(shape=[self.size, self.vocab_size], dtype=tf.float32))
        self.softmax_b = tf.Variable(random_uniform_initializer(shape=[self.vocab_size], dtype=tf.float32))
        
    def trainable_weights(self):
        return [
            *self.cells.trainable_weights,
            self.gate,
            self.atten_sum,
            self.u_f,
            self.u,
            self.w1,
            self.w2,
            self.b1,
            self.softmax_w,
            self.softmax_b
        ]

    def forward(self, input_data, init_output, mask, keyword_inputs, is_training=False):

        input_embeddings = tf.nn.embedding_lookup(params=self.embedding, ids=input_data)#返回一个tensor，shape是(batch_size, num_steps, size)
        keyword_embeddings = tf.nn.embedding_lookup(params=self.embedding, ids=keyword_inputs)

        # probably MTA part
        res1 = tf.Variable(tf.sigmoid(tf.matmul(tf.reshape(keyword_embeddings, [self.batch_size, -1]), self.u_f)))
        phi_res = tf.Variable(tf.reduce_sum(input_tensor=mask, axis=1, keepdims=True) * tf.cast(res1, tf.float32))
            
        self.output1 = phi_res
            
        outputs = []
        output_state = init_output.copy()
        state = self.cells.get_initial_state(batch_size = self.batch_size, dtype=tf.float32)
        # print(state.shape)
        # state = self._initial_state
        entropy_cost = []

        for time_step in range(self.num_steps):
            vs = []
            for s2 in range(config.num_keywords):
                a = tf.matmul(output_state, self.w1)
                b = tf.matmul(keyword_embeddings[:, s2, :], self.w2)
                c = tf.add(a, b)

                vi = tf.matmul(tf.tanh(tf.add(c, self.b1)), self.u)
                vs.append(vi * self.gate[:, s2:s2+1])
                
            attention_vs = tf.concat(vs, axis=1)
            prob_p = tf.nn.softmax(attention_vs)
            
            self.gate = tf.Variable(self.gate - (prob_p / phi_res))
            
            atten_sum = prob_p * mask[:, time_step:time_step+1]
            mt = tf.add_n([prob_p[:, i:i+1]*keyword_embeddings[:, i, :] for i in range(config.num_keywords)])

            # if time_step > 0: tf.compat.v1.get_variable_scope().reuse_variables()
            (cell_output, state) = self.cells(tf.concat([input_embeddings[:, time_step, :], mt], axis=1), state) 

            outputs.append(cell_output)
            output_state = cell_output
            
            end_output = cell_output
            
        self.output2 = atten_sum    
        output = tf.reshape(tf.concat(outputs, axis=1), [-1, self.size])
#         print("OUTPUT", output.shape)
#         print("SOFTMAX W", self.softmax_w.shape)
#         print("SOFTMAX B", self.softmax_b.shape)
        logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        _ , logits_dim = logits.shape
        logits = tf.reshape(logits, [self.batch_size, -1, logits_dim])
#         print("LOGITS", logits.shape)
#         print("TARGETS", self._targets.shape)
#         print("MASK", self._mask.shape)

#         if not is_training:
#             prob = tf.nn.softmax(logits)
#             self._sample = tf.argmax(input=prob, axis=1)
#             return
        return logits, phi_res, atten_sum


    def loss_calc(self, logits, targets, mask, phi_res, atten_sum):
        # Loss calculation
        loss = sequence_loss(
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
                'input_data': tf.io.FixedLenFeature([batch_size*num_steps], tf.int64),
                'target': tf.io.FixedLenFeature([batch_size*num_steps], tf.int64),
                'mask': tf.io.FixedLenFeature([batch_size*num_steps], tf.float32),
                'key_words': tf.io.FixedLenFeature([batch_size*config.num_keywords], tf.int64)
            }
        )
    train_dataset = tf.data.TFRecordDataset("coverage_data").map(decode_fn).shuffle(64).repeat(None)
    model = LSTMModel(is_training=True, config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        init_output = np.zeros((model.batch_size, model.size), dtype=np.float32)
        # Iterate over the batches of the dataset.
        for step, batch_dict in enumerate(tqdm(train_dataset)):
            input_data = batch_dict['input_data']
            target = batch_dict['target']
            mask = batch_dict['mask']
            key_words = batch_dict['key_words']
            
            input_data = tf.cast(input_data, tf.int32)
            target = tf.cast(target, tf.int32)
            mask = tf.cast(mask, tf.float32)
            key_words = tf.cast(key_words, tf.int32)

            input_data = tf.reshape(input_data, [batch_size, -1])
            target = tf.reshape(target, [batch_size, -1])
            mask = tf.reshape(mask, [batch_size, -1])
            key_words = tf.reshape(key_words, [batch_size, -1])
            
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                # logits = model(x_batch_train, training=True)  # Logits for this minibatch
                logits, phi_res, atten_sum = model.forward(input_data, init_output, mask, key_words, is_training=True)

                # Compute the loss value for this minibatch.
                loss_value = model.loss_calc(logits, target, mask, phi_res, atten_sum)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
#             for tw in model.trainable_weights():
#                 print(type(tw))
            grads = tape.gradient(loss_value, model.trainable_weights())

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights()))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
            
if __name__ == "__main__":
    main()