import torch
import logging
import argparse

from transformers.modeling_bert import BertForPretraining,load_tf_weights_in_bert
from transformers.configuration_bert import BertConfig

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path,bert_config_file,pytorch_dump_path):
  config = BertConfig.from_json_file(bert_config_file)
  model = BertForPretraining(config)
  
  load_tf_weights_in_bert(model,config,tf_checkpoint_path)
  
  torch.save(model.state_dict(),pytorch_dump_path)
  
if __name__ = '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--tf_checkpoint_path',default=None,type=str,required=True,
                      help='Path to the Tensorflow checkpoint path')
  
  parser.add_argument('--bert_config_file',default = None,type=str,required=True,
                      help='The config json file corresponds to the pretrained BERT model')
  
  parser.add_argument('--pytorch_model_path',default = None,type=str,required=True,
                      help='Path to the pyotrch model')
  
  args = parser.parse_args()
  convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,args.bert_config_file,args.pytorch_dump_path)
