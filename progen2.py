from fileOperation import *
def ProGen(fileName):
    import torch
    import numpy as np

    from tokenizers import Tokenizer
    from progen.progen2.models.progen.modeling_progen import ProGenForCausalLM

    def create_tokenizer_custom(file):
        with open(file, 'r') as f:
            return Tokenizer.from_str(f.read())

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    sequences=splitFile(fileName)

    print('Start loading model. Depending on your device, it may take a few minutes to load, please be patient.')
    model = ProGenForCausalLM.from_pretrained('/home/wangyansong/zilong/progen/progen2/checkpoints/progen2-base')
    tokenizer = create_tokenizer_custom(file='/home/wangyansong/zilong/progen/progen2/tokenizer.json')
    print('Model loading complete!')

    model = model.to(device)
    model = model.eval()
    print('Start generating features')  # 在默认情况下，PyTorch的模型是处于训练模式的。但如果我们需要对模型进行评估，那么就必须加上model.eval()来确保模型处于正确的模式下。
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
    print("features.shape:",features.shape)
    np.save('/home/wangyansong/wangyubo/PPI/seq_2_progen.npy', features)

