import os
import json
import sys
#sys.path.insert(0,'../')
sys.path.insert(1, os.path.join(sys.path[0],'..'))
from util import helper
from bert_utils import *
from bert import optimization
from util.metrics import *



class SelectionModel(object):
    """Defines the Tensorflow graph for training and testing a model."""

    def __init__(self, params, training_mode=True, optimizer=tf.train.AdamOptimizer, learning_rate=0.001):
        self.params = params
        opt = optimizer(learning_rate)
        self.BuildGraph(params, training_mode=training_mode, optimizer=opt)
        if not training_mode:
            self.BuildDecoderGraph()

    def BuildGraph(self, params, training_mode=True, optimizer=None):

        with tf.variable_scope('Selection'):

            self.prob_threshold = tf.placeholder_with_default(0.5, (), name='prob_threshold')
            self.dropout = tf.placeholder_with_default(0.9, (), name = 'dropout')
            self.input_ids = tf.placeholder(tf.int32, [None, params.max_seq_len], name='input_ids')
            self.input_mask = tf.placeholder(tf.int32, [None,params.max_seq_len], name='input_mask')
            self.segment_ids = tf.placeholder(tf.int32, [None,params.max_seq_len ], name='segment_ids')
            self.labels = tf.placeholder(tf.float32, [None,params.num_labels ], name='labels')
            self.bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
                                          trainable=True)

            #BERT instance
            bert_output = self.bert_module(
                dict(input_ids=self.input_ids, input_mask=self.input_mask,segment_ids=self.segment_ids),
                signature='tokens', as_dict=True)

            #Pooled Output from BERT
            output_layer = bert_output["pooled_output"]
            hidden_size = output_layer.shape[-1].value

            # Create our own layer to tune for politeness data.
            output_weights = tf.get_variable(
                "output_weights", [params.num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [params.num_labels], initializer=tf.zeros_initializer())

            with tf.variable_scope('loss'):

                output_layer = tf.nn.dropout(output_layer, keep_prob=self.dropout)

                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)

                sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels, logits = logits )

                # If we're train/eval, compute loss between predicted and actual label
                total_loss = tf.reduce_sum(sigmoid_loss)
                self.avg_loss = tf.to_float(total_loss)/tf.to_float(params.batch_size)


                self.probs = tf.nn.sigmoid(logits)
                self.predicted_labels = prediction_threshold(self.probs, self.prob_threshold)
                self.tp = true_positives(self.labels ,self.predicted_labels)
                self.fp = false_positives(self.labels, self.predicted_labels)
                self.fn = false_negatives(self.labels, self.predicted_labels)


        if training_mode:
            #Fine tune bert
            if params.fine_tune:
                self.train_op = optimization.create_optimizer(loss=self.avg_loss,
                                                              init_lr = params.learning_rate,
                                                              num_train_steps=params.iters,
                                                              num_warmup_steps= int(params.iters*params.warmup),
                                                              use_tpu =False)
            else:
                #Freeze bert
                train_vars = [var for var in tf.trainable_variables() if
                                var.name in ['Selection/output_weights:0', 'Selection/output_bias:0']]
                self.train_op = optimizer.minimize(self.avg_loss, var_list=train_vars)


    def BuildDecoderGraph(self):

        self.n_items = tf.placeholder_with_default(1, (), name='item_size')
        self.selected_probs, self.selected = tf.nn.top_k(self.probs, self.n_items)



class MetaSelectionModel(object):
    """Helper class for loading models."""

    def __init__(self, expdir, tokenizer = None):
        self.expdir = expdir
        self.params = helper.GetParams(os.path.join(expdir, 'params.json'), 'eval', expdir)
        self.class_indx = json.load(open(os.path.join(expdir, 'class_indx.json'),'r'))
        self.indx_to_class = {self.class_indx[c]:c for c in self.class_indx}
        if not tokenizer:
            self.tokenizer = create_tokenizer_from_hub_module()

        # construct the tensorflow graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = SelectionModel(self.params, training_mode=False)

    def MakeSession(self, threads=8):
        """Create the session with the given number of threads."""
        config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                                intra_op_parallelism_threads=threads)
        with self.graph.as_default():
            self.session = tf.Session(config=config)

    def Restore(self):
        """Initialize all variables and restore model from disk."""
        with self.graph.as_default():
            saver = tf.train.Saver(tf.trainable_variables())
            self.session.run(tf.global_variables_initializer())
            saver.restore(self.session, os.path.join(self.expdir, 'model.bin'))

    def MakeSessionAndRestore(self, threads=8):
        self.MakeSession(threads)
        self.Restore()


    def predict(self, query, threshold = 0.3, top_k = None):

        """Produce instance probabilities for a query"""

        input_features = convert_single_example(InputExample(guid=None,text_a=query,text_b=None, label=None),
                               self.params.max_seq_len, self.tokenizer)

        feed_dict = {
            self.model.input_ids: np.array(input_features.input_ids).reshape(1,-1),
            self.model.input_mask: np.array(input_features.input_mask).reshape(1,-1),
            self.model.segment_ids: np.array(input_features.segment_ids).reshape(1,-1),
            self.model.dropout: 1.0
        }

        if not top_k:
            feed_dict[self.model.n_items] = self.params.num_labels
            selected_probs, selected = self.session.run([self.model.selected_probs, self.model.selected], feed_dict)
            selected_classes = np.array([self.indx_to_class[i] for i in selected[0]])

            over_threshold = np.where(selected_probs[0]>threshold)[0]

            if over_threshold.size == 0:
                over_threshold = 0

            return selected_classes[over_threshold], selected_probs[0][over_threshold]

        else:
            feed_dict[self.model.n_items] = top_k
            selected_probs, selected = self.session.run([self.model.selected_probs, self.model.selected ], feed_dict)
            selected_classes = np.array([self.indx_to_class[i] for i in selected[0]])

            return selected_classes, selected_probs[0]