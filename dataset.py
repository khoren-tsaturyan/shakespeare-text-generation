import torch
from torch.utils import data


class ShakespeareDataset(data.Dataset):
    def __init__(self, tokenizer):
        super(ShakespeareDataset,self).__init__()
        DATA_PATH = "content/shakespeare.txt"
        full_text = self._get_data(DATA_PATH)
        full_text = '\n'.join([text.strip('"') for text in full_text.split('\n')])
        self.input_ids = []
        self.attn_masks = []
        sents = full_text.split('.')
        max_length = max([len(tokenizer.encode(sent)) for sent in sents])+2

        for sent in sents:
            sent = '<|startoftext|>' + sent + '<|endoftext|>'
            encoding = tokenizer(sent, padding="max_length",max_length=max_length)
            encoded = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            self.input_ids.append(torch.Tensor(encoded))
            self.attn_masks.append(torch.Tensor(attention_mask))
            
        self.dataset_size = len(self.input_ids)


    @staticmethod
    def _get_data(path):
        with open(path) as f:
            shakespeare_text = f.read()
        return shakespeare_text


    def __len__(self):
        return self.dataset_size


    def __getitem__(self,idx):
        return self.input_ids[idx], self.attn_masks[idx]
        

    
