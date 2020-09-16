CURRENT_DIR='pwd'

python $CURRENT_DIR/bert-master/create_pretraining_data.py \
--input_file=$CURRENT_DIR/raw_data/dataset.txt \
--output_file=$CURRENT_DIR/pretrain_data/tf_examples.tfrecord \
--vocab_file=$CURRENT_DIR/tf_model/vocab.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factors=5
