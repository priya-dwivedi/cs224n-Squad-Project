
from __future__ import print_function
import os
from collections import Counter
import string
import io

cwd = os.getcwd()

import json
import sys

### Usage python ensembly.py bidaf_json rnet_json output_json

def ensemble(preds_bidaf, preds_rnet):
    new_predictions = {}
    for key, value in preds_bidaf.iteritems():
        uuid = key
        ans_bidaf = value[0]
        prob_bidaf = value[1]
        if key in preds_rnet:  # find the same key
            val_rnet = preds_rnet[key]
            ans_rnet = val_rnet[0]
            prob_rnet = val_rnet[1]
        else:
            print("key not found in rnet_json")

        ## Check which prob is higher and then assign new answer
        if prob_bidaf >= prob_rnet:
            final_answer = ans_bidaf
        else:
            final_answer = ans_rnet

        new_predictions[uuid] = final_answer
    return new_predictions


file_bidaf = str(sys.argv[1])
file_rnet = str(sys.argv[2])
json_out_path = 'predictions.json'

# preds_bidaf = json.load(open(file_bidaf))
# preds_rnet = json.load(open(file_rnet))

preds_bidaf = json.load(open(os.path.join(cwd,file_bidaf)))
preds_rnet = json.load(open(os.path.join(cwd, file_rnet)))

new_predictions = ensemble(preds_bidaf, preds_rnet)

print("Writing predictions to %s..." % json_out_path)
with io.open(json_out_path, 'w', encoding='utf-8') as f:
    f.write(unicode(json.dumps(new_predictions, ensure_ascii=False)))
    print("Wrote predictions to %s" % json_out_path)