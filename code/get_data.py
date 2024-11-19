import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

label_list = ['Contact', 'Contact_Address_Book', 'Contact_City',
              'Contact_E_Mail_Address', 'Contact_Password', 'Contact_Phone_Number',
              'Contact_Postal_Address', 'Contact_ZIP', 'Demographic',
              'Demographic_Age', 'Demographic_Gender', 'Facebook_SSO',
              'Identifier', 'Identifier_Ad_ID', 'Identifier_Cookie_or_similar_Tech',
              'Identifier_Device_ID', 'Identifier_IMEI', 'Identifier_IMSI',
              'Identifier_IP_Address', 'Identifier_MAC', 'Identifier_Mobile_Carrier',
              'Identifier_SIM_Serial', 'Identifier_SSID_BSSID', 'Location',
              'Location_Bluetooth', 'Location_Cell_Tower', 'Location_GPS',
              'Location_IP_Address', 'Location_WiFi', 'SSO', 'No_Mentioned']

idx2label = {0: 'Contact', 1: 'Contact_Address_Book', 2: 'Contact_City',
             3: 'Contact_E_Mail_Address', 4: 'Contact_Password', 5: 'Contact_Phone_Number',
             6: 'Contact_Postal_Address', 7: 'Contact_ZIP', 8: 'Demographic',
             9: 'Demographic_Age', 10: 'Demographic_Gender', 11: 'Facebook_SSO',
             12: 'Identifier', 13: 'Identifier_Ad_ID', 14: 'Identifier_Cookie_or_similar_Tech',
             15: 'Identifier_Device_ID', 16: 'Identifier_IMEI', 17: 'Identifier_IMSI',
             18: 'Identifier_IP_Address', 19: 'Identifier_MAC', 20: 'Identifier_Mobile_Carrier',
             21: 'Identifier_SIM_Serial', 22: 'Identifier_SSID_BSSID', 23: 'Location',
             24: 'Location_Bluetooth', 25: 'Location_Cell_Tower', 26: 'Location_GPS',
             27: 'Location_IP_Address', 28: 'Location_WiFi', 29: 'SSO', 30: 'No_Mentioned'}

label2idx = {'Contact': 0, 'Contact_Address_Book': 1, 'Contact_City': 2,
             'Contact_E_Mail_Address': 3, 'Contact_Password': 4, 'Contact_Phone_Number': 5,
             'Contact_Postal_Address': 6, 'Contact_ZIP': 7, 'Demographic': 8,
             'Demographic_Age': 9, 'Demographic_Gender': 10, 'Facebook_SSO': 11,
             'Identifier': 12, 'Identifier_Ad_ID': 13, 'Identifier_Cookie_or_similar_Tech': 14,
             'Identifier_Device_ID': 15, 'Identifier_IMEI': 16, 'Identifier_IMSI': 17,
             'Identifier_IP_Address': 18, 'Identifier_MAC': 19, 'Identifier_Mobile_Carrier': 20,
             'Identifier_SIM_Serial': 21, 'Identifier_SSID_BSSID': 22, 'Location': 23,
             'Location_Bluetooth': 24, 'Location_Cell_Tower': 25, 'Location_GPS': 26,
             'Location_IP_Address': 27, 'Location_WiFi': 28, 'SSO': 29, 'No_Mentioned': 30}

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8

def read_data(file):
    """
    read data by file lines
    """
    texts = []      #text:<class 'str'>
    labels = []     #label:<class 'list'>

    for line in file.readlines():
        dic = json.loads(line)
        texts.append(dic['text'])
        # labels.append(dic['label'])
        label = []
        for item in dic['label']:
            label.append(label2idx[item])
        labels.append(label)

    return texts, labels

def get_train_and_val_data():
    """
    get train and validation data
    """
    with open('data/train.json', 'r', encoding='utf-8') as file:
        train_texts, train_labels = read_data(file)

    with open('data/valid.json', 'r', encoding='utf-8') as file:
        val_texts, val_labels = read_data(file) 

    # print(train_texts[:10])
    # print(train_labels[:10])
    # print(val_texts[:10])
    # print(val_labels[:10])
    print(len(train_labels), len(val_labels))   #10150 2817

    train_labels = one_hot_encode(train_labels, 31)    
    val_labels = one_hot_encode(val_labels, 31)    

    return train_texts, train_labels, val_texts, val_labels
  

def one_hot_encode(lst, length):
    """
    one hot encoding
    """
    result = []
    for sublist in lst:
        sub_result = np.zeros(length)
        for item in sublist:
            if 0 <= item < length:
                sub_result[item] = 1
        result.append(sub_result)
    return np.array(result)


class MyDataset(Dataset):
   
    def __init__(self, texts, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = texts
        self.targets = labels
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
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def get_DataLoader():
    model_name = 'code/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, do_lower_case=True)
    
    train_texts, train_labels, val_texts, val_labels = get_train_and_val_data()

    training_set = MyDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_set = MyDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4
                    }
    val_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 4
                }

    print("Create train and val DataLoader ing.")
    train_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    print("Create train and val DataLoader successfully.")

    return train_loader, val_loader