"""
Implementation of a QA System using AttentionWrapper.
"""

import numpy as np
import tensorflow as tf

# can be edited (to anything larger than vocab size) if encoding of vocab already uses 0, 1
END_TOKEN = 0
START_TOKEN = 1


class AttentionQA(object):
    def __init__(self, n_pointers=1, batch_size=150, seq_length=45, q_length = 5, learning_rate=0.0001,
                 cell=tf.contrib.rnn.GRUCell, n_layers=3, n_units=75, drop_out = 0.2):
        """Creates TensorFlow graph of a pointer network.

        Args:
            n_pointers (int):      Number of pointers to generate.
            batch_size (int) :     Batch size for training/inference.
            seq_length (int):      Maximum sequence length of inputs to encoder.
            learning_rate (float): Learning rate for Adam optimizer.
            cell (method):         Method to create single RNN cell.
            n_layers (int):        Number of layers in RNN (assumed to be the same for encoder & decoder).
            n_units (int):         Number of units in RNN cell (assumed to be the same for all cells).
        """

        with tf.variable_scope('inputs'):
            # integer-encoded input passages (e.g. 'She went home' -> [2, 3, 4])
            self.encoder_inputs = tf.placeholder(tf.int32, [batch_size, seq_length])
            # actual non-padded length of each input passages; used for dynamic unrolling
            # (e.g. ['She went home', 'She went to the station'] -> [3, 5])
            self.input_lengths = tf.placeholder(tf.int32, [batch_size])

            # integer-encoded input question (e.g. 'Where is she ?' -> [5, 6, 7, 8])
            self.question_inputs = tf.placeholder(tf.int32, [batch_size, q_length])
            # actual non-padded length of each input question; used for dynamic unrolling
            # (e.g. ['where is she ?', 'where is the dog ?'] -> [4, 5])
            self.question_lengths = tf.placeholder(tf.int32, [batch_size])

        with tf.variable_scope('outputs'):
            # pointer(s) to answer: (e.g. 'She went home' -> [2])
            self.pointer_labels = tf.placeholder(tf.int32, [batch_size, n_pointers])
            start_tokens = tf.constant(START_TOKEN, shape=[batch_size], dtype=tf.int32)
            # outputs of decoder are the word 'pointed' to by each pointer
            self.decoder_labels = tf.stack([tf.gather(inp, ptr) for inp, ptr in
                                           list(zip(tf.unstack(self.encoder_inputs), tf.unstack(self.pointer_labels)))])
            # inputs to decoder are inputs shifted over by one, with a <start> token at the front
            self.decoder_inputs = tf.concat([tf.expand_dims(start_tokens, 1), self.decoder_labels], 1)
            # output lengths are equal to the number of pointers
            self.output_lengths = tf.constant(n_pointers, shape=[batch_size])

        with tf.variable_scope('embeddings'):
            # load pre-trained GloVe embeddings
            word_matrix = tf.constant(np.load('./data/word_matrix.npy'), dtype=tf.float32)
            self.word_matrix = tf.Variable(word_matrix, trainable=True, name='word_matrix')
            # lookup embeddings of inputs & decoder inputs
            self.input_embeds = tf.nn.embedding_lookup(self.word_matrix, self.encoder_inputs)
            self.question_embeds = tf.nn.embedding_lookup(self.word_matrix, self.question_inputs)
            self.output_embeds = tf.nn.embedding_lookup(self.word_matrix, self.decoder_inputs)

        with tf.variable_scope('q_contexualize'):
            if n_layers > 1:
                q_cell_f = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell(n_units), output_keep_prob=1.0-drop_out) for _ in range(n_layers)])
                q_cell_b = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell(n_units), output_keep_prob=1.0-drop_out) for _ in range(n_layers)])
            else:
                q_cell_f = cell(n_units)
                q_cell_f = tf.contrib.rnn.DropoutWrapper(q_cell_f, output_keep_prob=1.0-drop_out)
                q_cell_b = cell(n_units)
                q_cell_b = tf.contrib.rnn.DropoutWrapper(q_cell_b, output_keep_prob=1.0-drop_out)
            q_contex, states = tf.nn.bidirectional_dynamic_rnn(q_cell_f,q_cell_b,self.question_embeds, self.question_lengths, dtype=tf.float32)
            self.q_contex = tf.concat(q_contex, axis=2)

        with tf.variable_scope('p_contexualize'):
            if n_layers > 1:
                p_cell_f = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell(n_units), output_keep_prob=1.0-drop_out) for _ in range(n_layers)])
                p_cell_b = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell(n_units), output_keep_prob=1.0-drop_out) for _ in range(n_layers)])
            else:
                p_cell_f = cell(n_units)
                p_cell_f = tf.contrib.rnn.DropoutWrapper(p_cell_f, output_keep_prob=1.0-drop_out)
                p_cell_b = cell(n_units)
                p_cell_b = tf.contrib.rnn.DropoutWrapper(p_cell_b, output_keep_prob=1.0-drop_out)
            p_contex, states = tf.nn.bidirectional_dynamic_rnn(p_cell_f, p_cell_b, self.input_embeds, self.input_lengths, dtype=tf.float32)
            self.p_contex = tf.concat(p_contex, axis=2)

        with tf.variable_scope('qa_alignment'):
            # create a soft-alignment of passage context given question
            attention = tf.contrib.seq2seq.BahdanauAttention(n_units, self.question_embeds,
            memory_sequence_length=self.question_lengths,dtype=tf.float32)
            # create an RNN over passage
            if n_layers > 1:
                qa_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell(n_units), output_keep_prob=1.0-drop_out) for _ in range(n_layers)])
            else:
                qa_cell = cell(n_units)
                qa_cell = tf.contrib.rnn.DropoutWrapper(qa_cell, output_keep_prob=1.0-drop_out)
            # for each input to the next RNN cell, wire the attention mechanism
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(qa_cell, attention, alignment_history=True)
            # define the initial state
            attention_state = attention_cell.zero_state(batch_size, dtype=tf.float32)

            helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.input_embeds, sequence_length=self.input_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, attention_state)
            self.qa_outputs, self.qa_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=seq_length)

            self.qa_alignment = tf.transpose(self.qa_state.alignment_history.stack(), [1,0,2])

        with tf.variable_scope('higher_qa_alignment'):
            # create a soft-alignment of passage context given question
            attention = tf.contrib.seq2seq.BahdanauAttention(n_units, self.q_contex,
            memory_sequence_length=self.question_lengths,dtype=tf.float32)
            # create an RNN over passage
            if n_layers > 1:
                hqa_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell(n_units), output_keep_prob=1.0-drop_out) for _ in range(n_layers)])
            else:
                hqa_cell = cell(n_units)
                hqa_cell = tf.contrib.rnn.DropoutWrapper(hqa_cell, output_keep_prob=1.0-drop_out)
            # for each input to the next RNN cell, wire the attention mechanism
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(hqa_cell, attention, alignment_history=True)
            # define the initial state
            attention_state = attention_cell.zero_state(batch_size, dtype=tf.float32)

            helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.input_embeds, sequence_length=self.input_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, attention_state)
            self.hqa_output, self.hqa_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=seq_length)

            self.hqa_alignment = tf.transpose(self.hqa_state.alignment_history.stack(), [1,0,2])

        # with tf.variable_scope('qq_alighment'):
        #     # attend over both original and qa aligned passage
        #     qq_input = tf.concat([self.input_embeds, self.qa_outputs.rnn_output], axis = 1)
        #     attention = tf.contrib.seq2seq.BahdanauAttention(n_units, qq_input,
        #     memory_sequence_length=self.question_lengths,dtype=tf.float32)
        #     # # create an RNN over passage
        #     if n_layers > 1:
        #         qq_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell(n_units), output_keep_prob=1.0-drop_out) for _ in range(n_layers)])
        #     else:
        #         qq_cell = cell(n_units)
        #         qq_cell = tf.contrib.rnn.DropoutWrapper(qq_cell, output_keep_prob=1.0-drop_out)
        #     # for each input to the next RNN cell, wire the attention mechanism
        #     attention_cell = tf.contrib.seq2seq.AttentionWrapper(qq_cell, attention, alignment_history=True)
        #     # define the initial state
        #     attention_state = attention_cell.zero_state(batch_size, dtype=tf.float32)
        #
        #     helper = tf.contrib.seq2seq.TrainingHelper(inputs=qq_input, sequence_length=self.input_lengths)
        #     decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, attention_state)
        #     self.qq_outputs, self.qq_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=seq_length)
        #
        #     self.qq_alignment = tf.transpose(self.qq_state.alignment_history.stack(), [1,0,2])

        with tf.variable_scope('encoder'):
            ptr_input = tf.concat([self.qa_outputs.rnn_output, self.hqa_output.rnn_output], axis = 2)
            if n_layers > 1:
                enc_cell = tf.contrib.rnn.MultiRNNCell([cell(n_units) for _ in range(n_layers)])
            else:
                enc_cell = cell(n_units)
            self.encoder_outputs, _ = tf.nn.dynamic_rnn(enc_cell, ptr_input, self.input_lengths, dtype=tf.float32)

        with tf.variable_scope('attention'):
            attention = tf.contrib.seq2seq.BahdanauAttention(n_units, self.encoder_outputs,
                                                             memory_sequence_length=self.input_lengths)

        with tf.variable_scope('decoder'):
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.output_embeds, sequence_length=self.output_lengths)
            if n_layers > 1:
                dec_cell = tf.contrib.rnn.MultiRNNCell([cell(n_units) for _ in range(n_layers)])
            else:
                dec_cell = cell(n_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention, alignment_history=True)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, word_matrix.shape[0]-2)
            decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, helper, out_cell.zero_state(batch_size, tf.float32))
            self.decoder_outputs, dec_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=n_pointers, impute_finished=True)

        with tf.variable_scope('pointers'):
            # tensor of shape (# pointers, batch size, max. input sequence length)
            self.pointer_prob = tf.reshape(dec_state.alignment_history.stack(), [n_pointers, batch_size, -1])
            self.pointers = tf.unstack(tf.argmax(self.pointer_prob, axis=2, output_type=tf.int32))

        with tf.variable_scope('loss'):
            loss = tf.zeros(())
            pointers = tf.unstack(self.pointer_prob)
            labels = tf.unstack(self.pointer_labels, axis=1)

            equal = []
            for i in range(n_pointers):
                loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[i], logits=pointers[i])
                equal.append(tf.equal(self.pointers[i], labels[i]))
            self.loss = tf.reduce_mean(loss)
            self.correct = tf.cast(tf.stack(equal), tf.float32)
            self.all_correct = tf.cast(tf.equal(tf.reduce_sum(self.correct, axis=0), n_pointers), tf.float32)
            self.exact_match = tf.reduce_mean(self.all_correct)

        with tf.variable_scope('training'):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


if __name__ == '__main__':
    m = AttentionQA()
