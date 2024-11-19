import torch
from transformers import AutoModel

LABELS_LEN = 31

class BertClass(torch.nn.Module):
    def __init__(self):
        super(BertClass, self).__init__()
        self.l1 = AutoModel.from_pretrained('code/bert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, LABELS_LEN)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        
        return output
    