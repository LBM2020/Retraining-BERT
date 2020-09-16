CURRENT_DIR='pwd'

python $CURRENT_DIR/tf2torch/tf2torch.py \
--tf_checkpoint_path=$CURRENT_DIR/pretrain_model_output \
--bert_config_file=$CURRENT_DIR/tf_model/bert_config.json \
--pytorch_model_path=$CURRENT_DIR/torch_model/pytorch_model.bin
