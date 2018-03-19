# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks, scopename):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(scopename):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class CNNEncoder(object):
    """
    Use CNN to generate encodings for each sentence
    """

    def __init__(self, num_filters, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.num_filters = num_filters
        self.keep_prob = keep_prob
        self.kernel_sizes = [2,3,4,5,6,7]


    ## Helper function to create conv1d layer

    def conv1d(input_, output_size, width, stride, scope):
        '''
        :param input_: A tensor of embedded tokens with shape [batch_size,max_length,embedding_size]
        :param output_size: The number of feature maps we'd like to calculate
        :param width: The filter width
        :param stride: The stride
        :return: A tensor of the concolved input with shape [batch_size,max_length,output_size]
        '''
        inputSize = input_.get_shape()[
            -1]  # How many channels on the input (The size of our embedding for instance)

        # This is the kicker where we make our text an image of height 1
        input_ = tf.expand_dims(input_, axis=1)  # Change the shape to [batch_size,1,max_length,output_size]

        # Make sure the height of the filter is 1
        with tf.variable_scope(scope=scope, reuse=tf.AUTO_REUSE):
            filter_ = tf.get_variable("conv_filter", shape=[1, width, inputSize, output_size])

        # Run the convolution as if this were an image
        convolved = tf.nn.conv2d(input_, filter=filter_, strides=[1, 1, stride, 1], padding="VALID")
        # Remove the extra dimension, eg make the shape [batch_size,max_length,output_size]
        result = tf.squeeze(convolved, axis=1)
        return result

    def build_graph(self, inputs, vec_len, scope_name):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        conv_outputs = []
        input_shape = inputs.get_shape().as_list()  # shape - batch_size, context/ques_len, embedding_dim
        inputs_expanded = tf.expand_dims(inputs, 1)  # becomes - batch_size, 1, context/ques_len, embedding_dim
        print("Shape before convolution ", inputs_expanded.shape)

        # with vs.variable_scope(scope=scope_name):
        conv_outputs = []
        for i, filter_size in enumerate(self.kernel_sizes):
            print("filter size is: ", filter_size)
            with tf.name_scope("conv-encoder-%s" % filter_size):
                # Convolution Layer

                if i ==0: # first convolution
                    input_shape = inputs_expanded.get_shape().as_list()
                    filter_shape = [1, filter_size, input_shape[-1], self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        inputs_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv")
                    # Apply nonlinearity
                    conv = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
                    drop  = tf.nn.dropout(conv, self.keep_prob)
                    print("Shape after convolution ", drop.shape)
                     #pass the results of convolution again back in for the next kernel size
                    # No max poooling is done to preserve input size
                    conv_outputs.append(drop)
                else:  # repeat convolutions on conv
                    input_shape = conv.get_shape().as_list()
                    filter_shape = [1, filter_size, input_shape[-1], self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        conv,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv")
                    # Apply nonlinearity
                    conv = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
                    drop = tf.nn.dropout(conv, self.keep_prob)
                    print("Shape after convolution ", drop.shape)
                    # pass the results of convolution again back in for the next kernel size
                    # No max poooling is done to preserve input size
                    conv_outputs.append(drop)

        # Combine all the pooled features
        print("conv outputs", conv_outputs)
        num_filters_total = self.num_filters * len(self.kernel_sizes)  #
        h_concat = tf.concat(conv_outputs, 3)  # across the filter dimension
        result = tf.squeeze(h_concat, axis=1)  # remove the extra dimension

        print("Result shape", result.shape)
        result = tf.reshape(result, [-1, vec_len, num_filters_total])  # batch size, question or cont

            # h_drop = tf.nn.dropout(h_reshaped, self.keep_prob)

        return result

class BiDAF(object):
    """
    Module for bidirectional attention flow.

    """

    def __init__(self, keep_prob, vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          vec_size: size of the word embeddings. int
        """
        self.keep_prob = keep_prob
        self.vec_size = vec_size
        self.S_W = tf.get_variable('S_W', [vec_size*3], tf.float32,
            tf.contrib.layers.xavier_initializer())


    def build_graph(self, q, q_mask, c, c_mask):
        """
        Inputs:
          c: context matrix, shape: (batch_size, num_context_words, vec_size).
          c_mask: Tensor shape (batch_size, num_context_words).
            1s where there's real input, 0s where there's padding
          q: question matrix (batch_size, num_question_words, vec_size)
          q_mask: Tensor shape (batch_size, num_question_words).
            1s where there's real input, 0s where there's padding
          N = num_context_words
          M = Num_question_words
          vec_size = hidden_size * 2

        Outputs:
          output: Tensor shape (batch_size, N, vec_size*3).
            This is the attention output.
        """
        with vs.variable_scope("BiDAF"):

            # Calculating similarity matrix
            c_expand = tf.expand_dims(c,2)  #[batch,N,1,2h]
            print(c_expand)
            q_expand = tf.expand_dims(q,1)  #[batch,1,M,2h]
            print(q_expand)
            c_pointWise_q = c_expand * q_expand  #[batch,N,M,2h]
            print(c_pointWise_q)

            c_input = tf.tile(c_expand, [1, 1, tf.shape(q)[1], 1])
            q_input = tf.tile(q_expand, [1, tf.shape(c)[1], 1, 1])

            concat_input = tf.concat([c_input, q_input, c_pointWise_q], -1) # [batch,N,M,6h]
            print(concat_input)


            similarity=tf.reduce_sum(concat_input * self.S_W, axis=3)  #[batch,N,M]
            print(similarity)

            # Calculating context to question attention
            similarity_mask = tf.expand_dims(q_mask, 1) # shape (batch_size, 1, M)
            print(similarity_mask)
            _, c2q_dist = masked_softmax(similarity, similarity_mask, 2) # shape (batch_size, N, M). take softmax over q
            print(c2q_dist)

            # Use attention distribution to take weighted sum of values
            c2q = tf.matmul(c2q_dist, q) # shape (batch_size, N, vec_size)
            print(c2q)

            # Calculating question to context attention c_dash
            S_max = tf.reduce_max(similarity, axis=2) # shape (batch, N)
            print(S_max)
            _, c_dash_dist = masked_softmax(S_max, c_mask, 1) # distribution of shape (batch, N)
            print(c_dash_dist)
            c_dash_dist_expand = tf.expand_dims(c_dash_dist, 1) # shape (batch, 1, N)
            print(c_dash_dist_expand)
            c_dash = tf.matmul(c_dash_dist_expand, c) # shape (batch_size, 1, vec_size)
            print(c_dash)

            c_c2q = c * c2q # shape (batch, N, vec_size)
            print(c_c2q)

            c_c_dash = c * c_dash # shape (batch, N, vec_size)
            print(c_c_dash)

            # concatenate the output
            output = tf.concat([c2q, c_c2q, c_c_dash], axis=2) # (batch_size, N, vec_size * 3)
            print(output)


            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)
            print(output)

            return output


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            print("Basic attn keys", keys.shape)
            print("Basic attn values", values_t.shape)
            print("Basic attn logits", attn_logits.shape)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class Attention_Match_RNN(object):
    """Module for Gated Attention and Self Matching from paper - https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
      Apply gated attention recurrent network for both query-passage matching and self matching networks
        Based on the explanation in http://web.stanford.edu/class/cs224n/default_project/default_project_v2.pdf
    """

    def create_weights(self, size_in, size_out, name):
        return tf.get_variable(name = name, dtype=tf.float32, shape=(size_in, size_out),
                                initializer=tf.contrib.layers.xavier_initializer())

    def create_vector(self, size_in, name):
        return tf.get_variable(name=name, dtype=tf.float32, shape=(size_in),
                               initializer=tf.contrib.layers.xavier_initializer())


    def matrix_multiplication(self, mat, weight):
        # [batch_size, seq_len, hidden_size] * [hidden_size, p] = [batch_size, seq_len, p]

        mat_shape = mat.get_shape().as_list()   #shape - ijk
        weight_shape = weight.get_shape().as_list()  #shape -kl
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])  #reshape to batch_size, seq_len, p

    def __init__(self, keep_prob, hidden_size_encoder, hidden_size_qp, hidden_size_sm):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          inp_vec_size: size of the input vector
        """
        self.keep_prob = keep_prob
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_qp = hidden_size_qp
        self.hidden_size_sm = hidden_size_sm

        # For QP attention
        self.W_uQ = self.create_weights(2 * self.hidden_size_encoder, self.hidden_size_qp, name='W_uQ')
        self.W_uP = self.create_weights(2 * self.hidden_size_encoder,  self.hidden_size_qp, name='W_uP')
        self.W_vP = self.create_weights(self.hidden_size_qp, self.hidden_size_qp, name='W_vP')
        self.W_g_QP = self.create_weights(4 * self.hidden_size_encoder, 4 * self.hidden_size_encoder, name='W_g_QP')
        self.v_t = self.create_vector(self.hidden_size_qp, name = 'v_t')

        # For self attention
        self.W_vP_self = self.create_weights(self.hidden_size_qp, self.hidden_size_sm, name='W_vP_self')
        self.W_vP_hat_self = self.create_weights(self.hidden_size_qp, self.hidden_size_sm, name='W_vP_hat_self')
        self.W_g_self = self.create_weights(2*self.hidden_size_qp, 2*self.hidden_size_qp, name='W_g_self')
        self.v_t_self= self.create_vector(self.hidden_size_sm, name='v_t_self')

        self.QP_cell = tf.contrib.rnn.GRUCell(self.hidden_size_qp)  # initiate GRU cell
        self.QP_cell = tf.contrib.rnn.DropoutWrapper(self.QP_cell,
                                                     input_keep_prob=self.keep_prob)  # added dropout wrapper

        self.SM_fw = tf.contrib.rnn.GRUCell(self.hidden_size_sm)  # initiate GRU cell
        self.SM_fw = tf.contrib.rnn.DropoutWrapper(self.SM_fw,
                                                   input_keep_prob=self.keep_prob)  # added dropout wrapper

        self.SM_bw = tf.contrib.rnn.GRUCell(self.hidden_size_sm)  # initiate GRU cell
        self.SM_bw = tf.contrib.rnn.DropoutWrapper(self.SM_bw,
                                                   input_keep_prob=self.keep_prob)  # added dropout wrapper


    def build_graph_qp_matching(self, context_encoding, question_encoding, values_mask, context_mask, context_len, question_len):
        """
I       Implement question passage matching from R-Net
        """

        v_P = []
        for i in range(context_len):

            ## As in the paper
            u_Q = question_encoding  # [batch_size, q_length, 2*hidden_size_encoder]
            u_P = context_encoding   # [batch_size, context_length, 2*hidden_size_encoder]

            W_uQ_uQ = self.matrix_multiplication(u_Q,self.W_uQ) # [batch_size, q_len, hidden_size_qp]
            print("Shape W_uQ_cal", W_uQ_uQ.shape)


            cur_batch_size = tf.shape(context_encoding)[0]
            concat_u_Pi = tf.concat([tf.reshape(u_P[:, i, :], [cur_batch_size, 1, 2*self.hidden_size_encoder])] * question_len, 1)

            # u_Pi_slice = tf.reshape(u_P[:, i, :], [cur_batch_size, 1, 2 * self.hidden_size_encoder])

            print("Shape Concat", concat_u_Pi)
            W_uP_uPi = self.matrix_multiplication(concat_u_Pi, self.W_uP) # [batch_size, 1, hidden_size_qp]

            print("Shape W_uP_cal", W_uP_uPi.shape)  # [batch_size, 1, hidden_size_qp]

            if i== 0:
                tanh_qp = tf.tanh(W_uQ_uQ+W_uP_uPi) # [batch_size, q_length, hidden_size_qp]
            else:

                concat_v_Pi = tf.concat([tf.reshape(v_P[i-1], [cur_batch_size, 1, self.hidden_size_qp])] * question_len, 1)
                # v_Pi_slice = tf.reshape(v_P[i - 1], [cur_batch_size, 1, self.hidden_size_qp])

                W_vP_vPi = self.matrix_multiplication(concat_v_Pi, self.W_vP)
                print("Shape W_vP_cal", W_vP_vPi.shape)  # [batch_size, 1, hidden_size_qp]
                tanh_qp = tf.tanh(W_uQ_uQ + W_uP_uPi+ W_vP_vPi)

            print("Shape tanh", tanh_qp.shape)
            # Calculate si = vT*tanh
            s_i_qp = self.matrix_multiplication(tanh_qp, tf.reshape(self.v_t, [-1,1]))  # [batch_size, q_length, 1]
            print("Shape s_i", s_i_qp.shape)

            s_i_qp = tf.squeeze(s_i_qp, axis=2) # [batch_size, q_length]. Same shape as values Mask

            # print("Shape values mask", values_mask.shape)

            _, a_i_qp = masked_softmax(s_i_qp, values_mask, 1)  # [batch_size, q_length]
            print("Shape a_i_qp", a_i_qp.shape)

            a_i_qp = tf.expand_dims(a_i_qp, axis=1)  # [batch_size, 1,  q_length]
            c_i_qp = tf.reduce_sum(tf.matmul(a_i_qp, u_Q), 1)  # [batch_size, 2 * hidden_size_encoder]


            print("Shape c_i", c_i_qp.shape)

            # gate

            slice = u_P[:, i, :]

            print("Shape slice", slice)
            u_iP_c_i = tf.concat([slice, c_i_qp], 1)
            print("Shape u_iP_c_i", u_iP_c_i.shape)  # [batch_size, 4*hidden_size_encoder]

            g_i = tf.sigmoid(tf.matmul(u_iP_c_i, self.W_g_QP))
            print("Shape g_i", g_i.shape)  # batch_size, 4*hidden_size_encoder]


            u_iP_c_i_star = tf.multiply(u_iP_c_i, g_i) # batch_size, 4*hidden_size_encoder]

            print("Shape u_iP_c_i_star", u_iP_c_i_star.shape)

            self.QP_state = self.QP_cell.zero_state(batch_size=cur_batch_size, dtype=tf.float32)

            # QP_attention
            with tf.variable_scope("QP_attention"):
                if i > 0: tf.get_variable_scope().reuse_variables()
                output, self.QP_state = self.QP_cell(u_iP_c_i_star, self.QP_state)
                v_P.append(output)
        v_P = tf.stack(v_P, 1)
        v_P = tf.nn.dropout(v_P, self.keep_prob)
        print("Shape v_P", v_P.shape) # [batch_size, context_len, hidden_size_qp]

        return v_P



    def build_graph_sm_matching(self, context_encoding, question_encoding, values_mask, context_mask, context_len,
                                question_len, v_P):
        """
I       Implement self matching from R-Net
        """

        ## Start Self Matching
        sm = []
        u_Q = question_encoding  # [batch_size, q_length, 2*hidden_size_encoder]
        u_P = context_encoding  # [batch_size, context_length, 2*hidden_size_encoder]

        for i in range(context_len):
            W_vP_vPself = self.matrix_multiplication(v_P, self.W_vP_self)  # [batch_size, context_len, hidden_size_sm]

            print("Shape W_vP_vPself", W_vP_vPself.shape)

            cur_batch_size = tf.shape(v_P)[0]

            # slice_v_iP = tf.reshape(v_P[:, i, :], [cur_batch_size, 1, self.hidden_size_qp])

            concat_v_iP = tf.concat(
                [tf.reshape(v_P[:, i, :], [cur_batch_size, 1, self.hidden_size_qp])] * context_len, 1)
            W_vP_vPhat_self = self.matrix_multiplication(concat_v_iP,
                                                         self.W_vP_hat_self)  # [batch_size, 1, hidden_size_sm]

            print("Shape W_vP_vPhat_self", W_vP_vPhat_self.shape)

            tanh_sm = tf.tanh(W_vP_vPself + W_vP_vPhat_self)  # [batch_size, context_len, hidden_size_sm]

            print("Shape tanh", tanh_sm.shape)

            # Calculate si = vT*tanh
            s_i_sm = self.matrix_multiplication(tanh_sm,
                                                tf.reshape(self.v_t_self, [-1, 1]))  # [batch_size, context_len, 1]
            print("Shape S_i", s_i_sm.shape)

            s_i_sm = tf.squeeze(s_i_sm, axis=2)  # [batch_size, context_len]

            _, a_i_sm = masked_softmax(s_i_sm, context_mask, 1)  # [batch_size, context_len]

            print("Shape a_i_sm", a_i_sm.shape)  # [batch_size, context_len]

            a_i_sm = tf.expand_dims(a_i_sm, axis=1)
            c_i_sm = tf.reduce_sum(tf.matmul(a_i_sm, v_P), 1)  # [batch_size, hidden_size_qp]

            print("Shape c_i", c_i_sm.shape)

            # gate
            slice_vP = v_P[:, i, :]
            v_iP_c_i = tf.concat([slice_vP, c_i_sm], 1)  # [batch_size, 2*hidden_size_qp]
            print("Shape v_iP_c_i", v_iP_c_i.shape)

            g_i_self = tf.sigmoid(tf.matmul(v_iP_c_i, self.W_g_self))  # [batch_size, 2*hidden_size_qp]
            print("Shape g_i_self", g_i_self.shape)

            v_iP_c_i_star = tf.multiply(v_iP_c_i, g_i_self)

            print("Shape v_iP_c_i_star", v_iP_c_i_star.shape)  # [batch_size, 2*hidden_size_qp]

            sm.append(v_iP_c_i_star)
        sm = tf.stack(sm, 1)
        unstacked_sm = tf.unstack(sm, context_len, 1)


        self.SM_fw_state = self.SM_fw.zero_state(batch_size=cur_batch_size, dtype=tf.float32)
        self.SM_bw_state = self.SM_bw.zero_state(batch_size=cur_batch_size, dtype=tf.float32)

        with tf.variable_scope('Self_match') as scope:
            SM_outputs, SM_final_fw, SM_final_bw = tf.contrib.rnn.static_bidirectional_rnn(self.SM_fw, self.SM_bw,
                                                                                           unstacked_sm,
                                                                                           dtype=tf.float32)
            h_P = tf.stack(SM_outputs, 1)
        h_P = tf.nn.dropout(h_P, self.keep_prob)

        print("Shape h_P", h_P.shape)  # [batch_size, context_len, 2*hidden_size_sm]

        return h_P


class Answer_Pointer(object):
    """
    Implement Question Pooling and Answer Pointer from RNET - https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
    """

    def create_weights(self, size_in, size_out, name):
        return tf.get_variable(name=name, dtype=tf.float32, shape=(size_in, size_out),
                               initializer=tf.contrib.layers.xavier_initializer())

    def create_vector(self, size_in, name):
        return tf.get_variable(name=name, dtype=tf.float32, shape=(size_in),
                               initializer=tf.contrib.layers.xavier_initializer())

    def matrix_multiplication(self, mat, weight):
        # [batch_size, seq_len, hidden_size] * [hidden_size, p] = [batch_size, seq_len, p]

        mat_shape = mat.get_shape().as_list()  # shape - ijk
        weight_shape = weight.get_shape().as_list()  # shape -kl
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])  # reshape to batch_size, seq_len, p


    def __init__(self, keep_prob, hidden_size_encoder, question_len, hidden_size_attn):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size_encoder = hidden_size_encoder
        self.keep_prob = keep_prob
        self.hidden_size_attn = hidden_size_attn
        self.question_len = question_len

        ## Initializations for question pooling
        self.W_ruQ = self.create_weights(2 * self.hidden_size_encoder, self.hidden_size_encoder, name='W_ruQ')
        self.W_vQ = self.create_weights(self.hidden_size_encoder, self.hidden_size_encoder, name='W_vQ')

        ## Same size as question hidden
        self.W_VrQ = self.create_weights(self.question_len, self.hidden_size_encoder,
                                        name='W_VrQ')

        self.v_qpool = self.create_vector(self.hidden_size_encoder, name = 'v_qpool')


        ## Initializations for answer pointer
        self.W_hP = self.create_weights( self.hidden_size_attn, 2*self.hidden_size_encoder, name='W_hP')
        self.W_ha = self.create_weights(2 * self.hidden_size_encoder, 2*self.hidden_size_encoder, name='W_ha')

        self.v_ptr = self.create_vector(2*self.hidden_size_encoder, name='v_ptr')

        self.ans_ptr_cell = tf.contrib.rnn.GRUCell(2 * self.hidden_size_encoder)  # initiate GRU cell
        self.ans_ptr_cell = tf.contrib.rnn.DropoutWrapper(self.ans_ptr_cell,
                                                          input_keep_prob=self.keep_prob)  # added dropout wrapper

    def question_pooling(self,question_encoding, values_mask):
        ## Question Pooling as suggested in R-Net Paper

        u_Q = question_encoding

        W_ruQ_u_Q = self.matrix_multiplication(u_Q, self.W_ruQ)  # [batch_size, q_length, hidden_size_encoder]
        print("Shape W_ruQ_u_Q", W_ruQ_u_Q.shape)

        W_vQ_V_rQ = tf.matmul(self.W_VrQ, self.W_vQ) # [ q_length, hidden_size_encoder]

        print("Shape W_vQ_V_rQ pre stack", W_vQ_V_rQ.shape)

        cur_batch_size = tf.shape(u_Q)[0]
        W_vQ_V_rQ = tf.expand_dims(W_vQ_V_rQ, axis =0)


        # print("Shape W_vQ_V_rQ in  stack", W_vQ_V_rQ.shape)
        # W_vQ_V_rQ = tf.stack([W_vQ_V_rQ] * cur_batch_size, 0)  # So shape  [batch_size, q_length, hidden_size_encoder]

        print("Shape W_vQ_V_rQ post stack", W_vQ_V_rQ.shape)

        tanh_qpool = tf.tanh(W_ruQ_u_Q + W_vQ_V_rQ)  # [batch_size, q_length, hidden_size_encoder]

        s_i_qpool = self.matrix_multiplication(tanh_qpool,
                                            tf.reshape(self.v_qpool, [-1, 1]))  # [batch_size, q_len, 1]

        s_i_qpool = tf.squeeze(s_i_qpool, axis=2)  # [batch_size, q_length]. Same shape as values Mask

        # print("Shape values mask", values_mask.shape)

        _, a_i_qpool = masked_softmax(s_i_qpool, values_mask, 1)  # [batch_size, q_length]

        a_i_qpool = tf.expand_dims(a_i_qpool, axis=1)  # [batch_size, 1,  q_length]
        r_Q = tf.reduce_sum(tf.matmul(a_i_qpool, u_Q), 1)  # [batch_size, 2 * hidden_size_encoder]

        r_Q = tf.nn.dropout(r_Q, self.keep_prob)
        print(' shape of r_Q', r_Q.shape)  # [batch_size, 2 * hidden_size_encoder]
        return r_Q



    def build_graph_answer_pointer(self, context_encoding, question_encoding, values_mask, context_mask, context_len,
                                    question_len, attn_output):

         #Implement answer pointer as suggested in the R-Net paper

        h_P = attn_output
        r_Q = self.question_pooling(question_encoding,values_mask)  # [batch_size, 2 * hidden_size_encoder]

        h_a = None

        cur_batch_size = tf.shape(question_encoding)[0]

        # r_Q is initial hidden vector for the pointer

        p = []

        logits = []

        for i in range(2):
             W_hP_h_P = self.matrix_multiplication(h_P, self.W_hP)  # [batch_size, context_len, 2*hidden_size_encoder]

             if i == 0:
                 h_i1a = r_Q
             else:
                 h_i1a = h_a
             print(' Shape of h_t1a', h_i1a.shape)
             concat_h_i1a = tf.concat([tf.reshape(h_i1a, [cur_batch_size, 1, 2*self.hidden_size_encoder])] * context_len, 1)
             W_ha_h_i1a = self.matrix_multiplication(concat_h_i1a, self.W_ha)

             tanh_ptr = tf.tanh(W_hP_h_P + W_ha_h_i1a)  # [batch_size, context_len, 2*hidden_size_encoder]

             print("Shape tanh_ptr", tanh_ptr.shape)

             s_i_ptr = self.matrix_multiplication(tanh_ptr,
                                            tf.reshape(self.v_ptr, [-1, 1]))  # [batch_size, context_len, 1]

             s_i_ptr = tf.squeeze(s_i_ptr, axis=2)   # [batch_size, context_len]. Same shape as context Mask

             print("Shape s_i_ptr", s_i_ptr.shape)

             logits_ptr, a_i_ptr = masked_softmax(s_i_ptr, context_mask, 1)  # [batch_size, context_len]

             print("Shape a_i_ptr", a_i_ptr.shape, logits_ptr.shape)

             print("Shape h_P", h_P.shape) # [batch_size, context_len, 2*hidden_size_encoder]



             p.append(a_i_ptr)   #prob distribution for output
             logits.append(logits_ptr) # logits from softmax

             a_i_ptr = tf.expand_dims(a_i_ptr, axis=1)
             c_i_ptr = tf.reduce_sum(tf.matmul(a_i_ptr, h_P), 1)  # [batch_size,  hidden_size_attn]

             if i == 0:
                 self.ans_ptr_state = self.ans_ptr_cell.zero_state(batch_size=cur_batch_size,
                                                                   dtype=tf.float32)  # set initial state to zero
                 h_a, _ = self.ans_ptr_cell(c_i_ptr, self.ans_ptr_state)
                 # h_a = h_a[1]
                 print("shape of h_a: ", h_a.shape)

        return p, logits

def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
