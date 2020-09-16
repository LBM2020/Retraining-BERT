CURRENT_DIR='pwd'

python $CURRENT_DIR/bert-master/run_pretraining.py \
--input_file=$CURRENT_DIR/pretrain_data/tf_examples.tfrecord \
--output_dir=$CURRENT_DIR/pretrain_model_output \
--do_train=True \
--do_eval=True \ 
--bert_config_file=$CURRENT_DIR/tf_model/bert_config.json \
--init_checkpoint=$CURRENT_DIR/tf_model/bert_model.ckpt \
--train_batch_size=32 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=20 \
--num_warmup_steps=10 \
--learning_rate=2e-5


