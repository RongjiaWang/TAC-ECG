
# 将文本数据进行编码

import datetime
from transformers import BertTokenizer

def bert_text(data):
    # 本地加载BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('/Bio_ClinicalBERT')
    texts_sum = []

    for index, text in enumerate(data):
        print("对第", index, "个文本进行词嵌入...")
        # texts = []
        text_i = ""
        for i in range(len(text)):
            if str(text[i]):
                text_i += str(text[i])
                text_i += " "
        print(text_i)
        # tokenizer.encode_plus负责将每个文本转换为BERT输入格式
        inputs = tokenizer.encode_plus(
            text_i,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,
        )
        input = inputs['input_ids']
        texts_sum.append(input)

    return texts_sum
