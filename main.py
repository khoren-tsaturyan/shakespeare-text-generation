import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader 
from dataset import ShakespeareDataset
from transformers import GPT2LMHeadModel
import copy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2Tokenizer
import argparse





def get_data(tokenizer,batch_size,random_seed=42):
    torch.manual_seed(random_seed)
    dataset = ShakespeareDataset(tokenizer)
    test_size = int(len(dataset)*0.05)
    val_size = int(len(dataset)*0.05) 
    train_size = len(dataset) - val_size - test_size
    train_split, test_split, val_split = torch.utils.data.random_split(dataset,[train_size, test_size, val_size])

    train_sampler = SubsetRandomSampler(train_split.indices)
    test_sampler = SubsetRandomSampler(test_split.indices)
    val_sampler = SubsetRandomSampler(val_split.indices)

    dataloaders = {
        'train': DataLoader(dataset, sampler=train_sampler, batch_size=batch_size),
        'val': DataLoader(dataset,sampler=val_sampler,batch_size=batch_size),
        'test': DataLoader(dataset,sampler=test_sampler,batch_size=1)  
    }

    dataset_sizes = {'train': train_size, 'val':val_size, 'test': test_size}
    return dataloaders,dataset_sizes





def visualize_logs(logs,save_figure=False,save_dir=None):
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    ax1.plot(logs['Epoch'],logs['Train_loss'],label='train_loss')
    ax1.plot(logs['Epoch'],logs['Val_loss'],label='val_loss')
    ax1.legend(loc='best')
    ax1.set_title('Losses')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    fig.tight_layout(pad=1)
    

    if save_figure:
        fig.savefig(save_dir)
    
    plt.show()

def train(dataloaders,dataset_sizes,model,device,epochs,optimizer):
    best_loss = np.inf
    best_model = copy.deepcopy(model.state_dict())
    df = pd.DataFrame()
    train_loss = []
    val_loss = []
    for epoch in range(1,epochs+1):
        print(f'Epoch {epochs}/{epoch}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            for inputs,att_masks in tqdm(dataloaders[phase]):
                att_masks = att_masks.to(torch.int64).to(device)
                inputs = inputs.to(torch.int64).to(device)
                labels = inputs.clone().to(torch.int64).to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, labels=labels,attention_mask=att_masks)
                    loss, logits = outputs[:2]
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item()*batch_size

            epoch_loss = round(running_loss / dataset_sizes[phase],4)

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
                
            print('{} Loss: {}'.format(
                phase, epoch_loss),end='\n')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

    print(f'Best Val Loss.: {best_loss:.4f}, In Epoch: {best_epoch}')
    df['Epoch'] = range(1,epochs+1)
    df['Train_loss'] = train_loss
    df['Val_loss'] = val_loss
    df['Batch_Size'] = batch_size
    
    model.load_state_dict(best_model)
    
    return model,df



def main():
    parser = argparse.ArgumentParser(description='GPT2 Fine-Tuning')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    device = torch.device("cuda" if use_cuda else "cpu")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token='<|startoftext|>', 
                                              eos_token='<|endoftext|>', 
                                              pad_token='<|pad|>', pad_to_multiple_of=8)
    dataloaders,dataset_sizes = get_data(tokenizer,args.batch_size,random_seed=args.seed)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model,logs = train(dataloaders,dataset_sizes,model,device,args.epochs,optimizer)

    if args.save_model:
        logs.to_csv('logs/training_logs.csv',index=False)
        model.save_pretrained('results')
        tokenizer.save_pretrained('results')
        visualize_logs(logs,True,f'logs/training_plot.png')


    
if __name__=='__main__':
    main()
