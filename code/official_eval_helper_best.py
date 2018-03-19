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

"""This code is required for "official_eval" mode in main.py
It provides functions to read a SQuAD json file, use the model to get predicted answers,
and write those answers to another JSON file."""

from __future__ import absolute_import
from __future__ import division

import os
from tqdm import tqdm
import numpy as np
from six.moves import xrange
from nltk.tokenize.moses import MosesDetokenizer

from preprocessing.squad_preprocess import data_from_json, tokenize
from vocab import UNK_ID, PAD_ID
from data_batcher import padded, Batch



def readnext(x):
    """x is a list"""
    if len(x) == 0:
        return False
    else:
        return x.pop(0)



def refill_batches(batches, word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len):
    """
    This is similar to refill_batches in data_batcher.py, but:
      (1) instead of reading from (preprocessed) datafiles, it reads from the provided lists
      (2) it only puts the context and question information in the batches (not the answer information)
      (3) it also gets UUID information and puts it in the batches

    Inputs:
      batches: list to be refilled
      qn_uuid_data: list of strings that are unique ids
      context_token_data, qn_token_data: list of lists of strings (no UNKs, no padding)
      batch_size: int. size of batches to make
      context_len, question_len: ints. max sizes of context and question. Anything longer is truncated.

    Makes batches that contain:
      uuids_batch, context_tokens_batch, context_ids_batch, qn_ids_batch: all lists length batch_size
    """
    examples = []

    # Get next example
    qn_uuid, context_tokens, qn_tokens = readnext(qn_uuid_data), readnext(context_token_data), readnext(qn_token_data)

    while qn_uuid and context_tokens and qn_tokens:

        # Convert context_tokens and qn_tokens to context_ids and qn_ids
        context_ids = [word2id.get(w, UNK_ID) for w in context_tokens]
        qn_ids = [word2id.get(w, UNK_ID) for w in qn_tokens]

        # Truncate context_ids and qn_ids
        # Note: truncating context_ids may truncate the correct answer, meaning that it's impossible for your model to get the correct answer on this example!
        if len(qn_ids) > question_len:
            qn_ids = qn_ids[:question_len]
        if len(context_ids) > context_len:
            context_ids = context_ids[:context_len]

        # Add to list of examples
        examples.append((qn_uuid, context_tokens, context_ids, qn_ids))

        # Stop if you've got a batch
        if len(examples) == batch_size:
            break

        # Get next example
        qn_uuid, context_tokens, qn_tokens = readnext(qn_uuid_data), readnext(context_token_data), readnext(qn_token_data)

    # Make into batches
    for batch_start in xrange(0, len(examples), batch_size):
        uuids_batch, context_tokens_batch, context_ids_batch, qn_ids_batch = zip(*examples[batch_start:batch_start + batch_size])

        batches.append((uuids_batch, context_tokens_batch, context_ids_batch, qn_ids_batch))

    return



def get_batch_generator(word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len):
    """
    This is similar to get_batch_generator in data_batcher.py, but with some
    differences (see explanation in refill_batches).

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      qn_uuid_data: list of strings that are unique ids
      context_token_data, qn_token_data: list of lists of strings (no UNKs, no padding)
      batch_size: int. size of batches to make
      context_len, question_len: ints. max sizes of context and question. Anything longer is truncated.

    Yields:
      Batch objects, but they only contain context and question information (no answer information)
    """
    batches = []

    while True:
        if len(batches) == 0:
            refill_batches(batches, word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len)
        if len(batches) == 0:
            break

        # Get next batch. These are all lists length batch_size
        (uuids, context_tokens, context_ids, qn_ids) = batches.pop(0)

        # Pad context_ids and qn_ids
        qn_ids = padded(qn_ids, question_len) # pad questions to length question_len
        context_ids = padded(context_ids, context_len) # pad contexts to length context_len

        # Make qn_ids into a np array and create qn_mask
        qn_ids = np.array(qn_ids)
        qn_mask = (qn_ids != PAD_ID).astype(np.int32)

        # Make context_ids into a np array and create context_mask
        context_ids = np.array(context_ids)
        context_mask = (context_ids != PAD_ID).astype(np.int32)

        # Make into a Batch object
        batch = Batch(context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens=None, ans_span=None, ans_tokens=None, uuids=uuids)

        yield batch

    return


def preprocess_dataset(dataset):
    """
    Note: this is similar to squad_preprocess.preprocess_and_write, but:
      (1) We only extract the context and question information from the JSON file.
        We don't extract answer information. This makes this function much simpler
        than squad_preprocess.preprocess_and_write, because we don't have to convert
        the character spans to word spans. This also means that we don't have to
        discard any examples due to tokenization problems.

    Input:
      dataset: data read from SQuAD JSON file

    Returns:
      qn_uuid_data, context_token_data, qn_token_data: lists of uuids, tokenized context and tokenized questions
    """
    qn_uuid_data = []
    context_token_data = []
    qn_token_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing data"):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            context = unicode(article_paragraphs[pid]['context']) # string

            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context) # list of strings (lowercase)
            context = context.lower()

            qas = article_paragraphs[pid]['qas'] # list of questions

            # for each question
            for qn in qas:

                # read the question text and tokenize
                question = unicode(qn['question']) # string
                question_tokens = tokenize(question) # list of strings

                # also get the question_uuid
                question_uuid = qn['id']

                # Append to data lists
                qn_uuid_data.append(question_uuid)
                context_token_data.append(context_tokens)
                qn_token_data.append(question_tokens)

    return qn_uuid_data, context_token_data, qn_token_data


def get_json_data(data_filename):
    """
    Read the contexts and questions from a .json file (like dev-v1.1.json)

    Returns:
      qn_uuid_data: list (length equal to dev set size) of unicode strings like '56be4db0acb8001400a502ec'
      context_token_data, qn_token_data: lists (length equal to dev set size) of lists of strings (no UNKs, unpadded)
    """
    # Check the data file exists
    if not os.path.exists(data_filename):
        raise Exception("JSON input file does not exist: %s" % data_filename)

    # Read the json file
    print "Reading data from %s..." % data_filename
    data = data_from_json(data_filename)

    # Get the tokenized contexts and questions, and unique question identifiers
    print "Preprocessing data from %s..." % data_filename
    qn_uuid_data, context_token_data, qn_token_data = preprocess_dataset(data)

    data_size = len(qn_uuid_data)
    assert len(context_token_data) == data_size
    assert len(qn_token_data) == data_size
    print "Finished preprocessing. Got %i examples from %s" % (data_size, data_filename)

    return qn_uuid_data, context_token_data, qn_token_data


def generate_answers(session, model, word2id, qn_uuid_data, context_token_data, qn_token_data):
    """
    Given a model, and a set of (context, question) pairs, each with a unique ID,
    use the model to generate an answer for each pair, and return a dictionary mapping
    each unique ID to the generated answer.

    Inputs:
      session: TensorFlow session
      model: QAModel
      word2id: dictionary mapping word (string) to word id (int)
      qn_uuid_data, context_token_data, qn_token_data: lists

    Outputs:
      uuid2ans: dictionary mapping uuid (string) to predicted answer (string; detokenized)
    """
    uuid2ans = {} # maps uuid to string containing predicted answer
    data_size = len(qn_uuid_data)
    num_batches = ((data_size-1) / model.FLAGS.batch_size) + 1
    batch_num = 0
    detokenizer = MosesDetokenizer()

    print "Generating answers..."

    for batch in get_batch_generator(word2id, qn_uuid_data, context_token_data, qn_token_data, model.FLAGS.batch_size, model.FLAGS.context_len, model.FLAGS.question_len):

        # Get the predicted spans
        pred_start_batch, pred_end_batch = model.get_start_end_pos(session, batch)

        # Convert pred_start_batch and pred_end_batch to lists length batch_size
        pred_start_batch = pred_start_batch.tolist()
        pred_end_batch = pred_end_batch.tolist()

        # For each example in the batch:
        for ex_idx, (pred_start, pred_end) in enumerate(zip(pred_start_batch, pred_end_batch)):

            # Original context tokens (no UNKs or padding) for this example
            context_tokens = batch.context_tokens[ex_idx] # list of strings

            # Check the predicted span is in range
            assert pred_start in range(len(context_tokens))
            assert pred_end in range(len(context_tokens))

            # Predicted answer tokens
            pred_ans_tokens = context_tokens[pred_start : pred_end +1] # list of strings

            # Detokenize and add to dict
            uuid = batch.uuids[ex_idx]
            uuid2ans[uuid] = detokenizer.detokenize(pred_ans_tokens, return_str=True)

        batch_num += 1

        if batch_num % 10 == 0:
            print "Generated answers for %i/%i batches = %.2f%%" % (batch_num, num_batches, batch_num*100.0/num_batches)

    print "Finished generating answers for dataset."

    return uuid2ans
