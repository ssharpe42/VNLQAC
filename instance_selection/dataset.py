from bert_utils import *


def LoadData(filename, limit=None):
    """
    Load a queries and split into train test

    """
    query_classes = pd.read_csv(filename, sep='\t', index_col=None)

    if limit and isinstance(limit, int):
        query_classes = query_classes.sample(limit)

    # Get Unique classes and queries
    classes = np.unique(query_classes['class'])
    queries = np.unique(query_classes['query'])
    class_indx = {classes[i]: i for i in range(len(classes))}
    query_dict = {q: np.zeros(len(classes), dtype='int8') for q in queries}

    # Fill multi hot vectors
    for i in range(query_classes.shape[0]):

        if i % 100000 == 0:
            print('Setting class vectors: {:,}'.format(i))

        query = query_classes['query'].iloc[i]
        c = query_classes['class'].iloc[i]
        query_dict[query][class_indx[c]] = 1

    # Labels by index
    labels = sorted(class_indx, key=class_indx.get)

    return query_dict, class_indx, labels


class Dataset(object):

    def __init__(self, query_dict, query_set, tokenizer, num_labels, batch_size=32, max_seq_len=20):

        self.query_dict = query_dict
        self.query_set = query_set
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.max_seq_len = max_seq_len
        self.current_idx = 0

        np.random.shuffle(self.query_set)

        print('Loading input examples....')

        self.InputExamples = []
        for i in range(len(self.query_set)):
            if i % 100000 == 0:
                print('Appending {:,}'.format(i))
            q = self.query_set[i]
            self.InputExamples.append(InputExample(guid=None,
                                                   text_a=q,
                                                   text_b=None,
                                                   label=self.query_dict[q]))


    def GetFeedDict(self, model):

        if self.current_idx + self.batch_size > len(self.query_set):
            self.current_idx = 0

        idx = range(self.current_idx, self.current_idx + self.batch_size)
        self.current_idx += self.batch_size

        batch = [self.InputExamples[i] for i in idx]

        batch = [convert_single_example(ex, self.max_seq_len, self.tokenizer) for ex in batch]

        bert_dict = {
            'input_ids': np.zeros((self.batch_size, self.max_seq_len)),
            'input_mask': np.zeros((self.batch_size, self.max_seq_len)),
            'segment_ids': np.zeros((self.batch_size, self.max_seq_len)),
            'labels': np.zeros((self.batch_size, self.num_labels))
        }

        for i in xrange(len(batch)):
            sample = batch[i]
            bert_dict['input_ids'][i, :] = sample.input_ids
            bert_dict['input_mask'][i, :] = sample.input_mask
            bert_dict['segment_ids'][i, :] = sample.segment_ids
            bert_dict['labels'][i, :] = sample.labels

        feed_dict = {
            model.input_ids: bert_dict['input_ids'],
            model.input_mask: bert_dict['input_mask'],
            model.segment_ids: bert_dict['segment_ids'],
            model.labels: bert_dict['labels']
        }

        return feed_dict
