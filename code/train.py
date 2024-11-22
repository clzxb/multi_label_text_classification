import torch
from tqdm import tqdm
from model import BertClass
from get_data import get_DataLoader
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-05
model_dir = "code/finetune_models/my_finetune_model.pth"

def calculate_accuracy(preds, targets):
    n_correct = (preds == targets).all(dim=1).sum().item()
    return n_correct

def calculate_precision(preds, targets):
    count = 0
    for i in range(targets.shape[0]):
        if sum(preds[i]) == 0:
            continue
        count += sum(np.logical_and(targets[i], preds[i])) / sum(preds[i])
    return count / targets.shape[0]

def calculate_recall(preds, targets):
    count = 0
    for i in range(targets.shape[0]):
        if sum(targets[i]) == 0:
            continue
        count += sum(np.logical_and(targets[i], preds[i])) / sum(targets[i])
    return count / targets.shape[0]

def calculate_big_idx(outputs):
    new_outputs = []
    correct_threshold = 0.5
    for output in outputs:
        new_output = []
        for item in output:
            if item > correct_threshold:
                new_output.append(1)
            else:
                new_output.append(0)
        new_outputs.append(new_output)
    return torch.tensor(new_outputs)

def train(epoch, training_loader): 
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    precision = 0
    recall = 0

    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)        
    
        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs.float(), targets.float())
        tr_loss += loss.item()
        
        big_idx = calculate_big_idx(outputs)
        n_correct += calculate_accuracy(big_idx.to(device), targets)
        precision += calculate_precision(big_idx.cpu(), targets.cpu())
        recall += calculate_recall(big_idx.cpu(), targets.cpu())

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 100 steps: {loss_step}")
            print(f"Training Accuracy per 100 steps: {accu_step}")
            print(f"Training Precision per 100 steps: {precision * 100 / _}")
            print(f"Training Recall per 100 steps: {recall * 100 / _}")


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch + 1}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    print(f"Training Precision Epoch: {precision * 100 / _}")
    print(f"Training Recall Epoch: {recall * 100 / _}")
    return


def valid(model, val_loader):
    model.eval()

    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    precision = 0
    recall = 0

    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long) 

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs.float(), targets.float())
            tr_loss += loss.item()

            big_idx = calculate_big_idx(outputs)
            n_correct += calculate_accuracy(big_idx.to(device), targets)
            precision += calculate_precision(big_idx.cpu(), targets.cpu())
            recall += calculate_recall(big_idx.cpu(), targets.cpu())

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation Loss per 5000 steps: {loss_step}")
                print(f"Validation Accuracy per 5000 steps: {accu_step}")
                print(f"Validation Precision per 5000 steps: {precision * 100 / _}")
                print(f"Validation Recall per 5000 steps: {recall * 100 / _}")
   
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples

    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    print(f"Validation Precision Epoch: {precision * 100 / _}")
    print(f"Validation Recall Epoch: {recall * 100 / _}")

    return epoch_accu, precision * 100 / _, recall * 100 / _


if __name__ == '__main__':
    model = BertClass() #finetuning
    #model = torch.nn.DataParallel(model)
    model.to(device)

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_DataLoader()
    EPOCHS = 10

    acc = 0
    pre = 0
    rec = 0
    print("Training start.")
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1}")
        train(epoch, train_loader)
        acc_current, pre_current, rec_current= valid(model, val_loader)
        if acc_current > acc:
            acc = acc_current
            pre = pre_current
            rec = rec_current
            torch.save(model, model_dir)
            print("Save successfully.")

    print(f"Best Validation Accuracy: {acc}")
    print(f"Best Validation Precision: {pre}")
    print(f"Best Validation Recall: {rec}")