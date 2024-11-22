import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from train import calculate_big_idx
from get_data import idx2label
import re


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
model_dir = "code/finetune_models/my_finetune_model.pth"
model_name = 'code/bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, do_lower_case=True)

class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = texts
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text, 
            None,  
            add_special_tokens=True,
            max_length=self.max_len, 
            pad_to_max_length=True, 
            return_token_type_ids=True  
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
            #'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


def test(model, testing_loader):
    res = []
    model.eval()    

    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            #targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).to(device)

            big_idx = calculate_big_idx(outputs)
            res.extend(big_idx.tolist())            

    return res

if __name__ == '__main__':
    print('Reading test data.')
    with open('data/test.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    test_data = content.split('\n')
    test_data = list(filter(None, test_data))

    test_dataset = MyDataset(test_data, tokenizer, MAX_LEN)
    test_params = {'batch_size': 32, 
                'shuffle': False,
                'num_workers': 2
                }
    test_loader = DataLoader(test_dataset, **test_params)

    model = torch.load(model_dir, map_location="cpu") 
    model.to(device)
    res = test(model, test_loader)
    print(len(res))
    # get one-hot result
    with open('data/test_result.txt', 'w', encoding='utf-8') as file:
        for content in res:
            file.write("%s\n" % content)
    print('Save successfully.')

    # get real-label result
    real_result = []
    for item in res:
        sub_real_result = []
        for idx, value in enumerate(item):
            if value == 1:
                sub_real_result.append(idx2label[idx])
        real_result.append(sub_real_result)
    # print(len(real_result))
    # print(real_result)

    with open('data/test_real_result.txt', 'w', encoding='utf-8') as file:
            for content in real_result:
                content = ','.join(content)
                if content == '':
                    content = 'No_Mentioned'
                file.write("%s\n" % content)
    print('Save successfully.')