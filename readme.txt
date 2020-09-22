本工程目的在于在公开BERT模型的基础上使用自己的数据继续预训练，并且将预训练得到的tensorflow版本的BERT模型转换为pytorch版本的模型，文件执行顺序如下：
1) create_pretrain_data.sh  用于创建符合BERT预训练数据格式的数据
2）pretrain_model.sh        使用创建好的数据对BERT模型进行训练
3）tf2torch.sh              将训练得到的tf模型转换为pytorch模型
