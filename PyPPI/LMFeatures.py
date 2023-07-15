import os
import numpy as np
from fileOperation import *

fileName=os.environ.get('INPUT_FN')


def generateProgenFeatures():
    import torch
    from tokenizers import Tokenizer
    from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM

    def create_tokenizer_custom(file):
        with open(file, 'r') as f:
            return Tokenizer.from_str(f.read())

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    sequences = splitFile(fileName)

    print('Start loading model. Depending on your device, it may take a few minutes to load, please be patient.')
    model = ProGenForCausalLM.from_pretrained('/home/wangyansong/zilong/progen/progen2/checkpoints/progen2-base')
    tokenizer = create_tokenizer_custom(file='/home/wangyansong/zilong/progen/progen2/tokenizer.json')
    print('Model loading complete!')

    model = model.to(device)
    model = model.eval() # By default, PyTorch models are in training mode. But if we need to evaluate the model, we must add model.eval() to ensure that the model is in the correct mode.
    print('Start generating features')
    features = []
    i = 0
    for seq in sequences:
        i += 1
        if len(seq) >= 2000:
            seq = seq[:1000]
        target = torch.tensor(tokenizer.encode(seq).ids).to(device)

        logits = model(target).logits.cpu().detach().numpy()
        features.append(logits)

    features = np.concatenate(features)
    return features


def generateESM1bFeatures():
    fileName = os.environ.get('INPUT_FN')
    from transformers import ESMForMaskedLM, ESMTokenizer
    print('Start loading model. Depending on your device, it may take a few minutes to load, please be patient.')
    tokenizer = ESMTokenizer.from_pretrained("/home/wangyansong/zilong/esm-1b", do_lower_case=False)
    model = ESMForMaskedLM.from_pretrained("/home/wangyansong/zilong/esm-1b")
    print('Model loading complete!')
    # 整合时替换
    # tokenizer = ESMTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False )
    # model = ESMForMaskedLM.from_pretrained("facebook/esm-1b")
    import torch

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sequences = splitFile(fileName)
    seq_name = []
    sequences = []
    seq_file = open(fileName, 'r')
    for line in seq_file:
        if line.startswith('>'):
            seq_name.append(line[1:].strip())
        else:
            sequences.append(line.strip())

    model = model.to(device)
    model = model.eval()
    print('Start generating features, please be patient!')
    features = []
    for seq in sequences:
        ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        features.append(np.array(embedding['logits'].cpu()))
    features = np.concatenate(features)
    return features


def generateProtT5Features():
    from transformers import TFT5EncoderModel, T5Tokenizer, T5EncoderModel
    import re
    import gc

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    gc.collect()
    print("It takes time to load the model, please be patient.")
    model = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", from_pt=True)
    print("The model is loaded already.")

    sequences_Example = splitFile(fileName)

    seq_split = []
    seq_split.append(str(' '.join([word for word in sequences_Example])))
    sequences_Example = seq_split

    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True, return_tensors="tf")

    input_ids = ids['input_ids']
    attention_mask = ids['attention_mask']
    embedding = model(input_ids)
    embedding = np.asarray(embedding.last_hidden_state)
    attention_mask = np.asarray(attention_mask)

    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    features = np.array(features)
    return features