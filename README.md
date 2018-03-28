## cs224n-Squad-Project

This repository has the code to run the model my team built for the SQUaD dataset

### Running the code

* Please run ```./get_started.sh``` to download the SQuAD dataset and GloVE Vectors
* requirements.txt is used by get_started.sh to install requirements. Once the script is done running, you will have a new directory data with the train and dev json files for SQuAD datset. And another empty folder experiments that will eventually have the results from your experiments. 

* To run code please run main.py in code. The settings to run BIDAF model are:
```
python code/main.py --experiment_name=bidaf_best --dropout=0.15 --batch_size=60 --hidden_size_encoder=150 --embedding_size=100 --do_char_embed=False --add_highway_layer=True --rnet_attention=False --bidaf_attention=True --answer_pointer_RNET=False --smart_span=True --hidden_size_modeling=150 --mode=train
```

* The settings to run the RNET model are:

```
python code/main.py --experiment_name=rnet_best --dropout=0.20 --batch_size=20 --hidden_size_encoder=200 --embedding_size=300 --do_char_embed=False --add_highway_layer=False --rnet_attention=True --bidaf_attention=False --answer_pointer_RNET=True --smart_span=True--mode=official_eval \
--json_in_path=data/tiny-dev.json \
--json_out_path=predictions_rnet.json \
--ckpt_load_dir=experiments/rnet_best/best_checkpoint
```

* Once you run the models, you will have a new folder by the name experiments which will have the results from your code runs

* To start tensorboard, please run the following commands:
```
cd experiments # Go to experiments directory
tensorboard --logdir=. --port=5678 # Start TensorBoard
```
