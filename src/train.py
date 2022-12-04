import torch
from torch import nn
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

from src.models import MULTModel
from src.utils import *
from src.eval_metrics import *

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = MULTModel(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    print(settings)
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

def get_eval_metrics(hyp_params, results, truths, wandb_logging=False):
    if hyp_params.dataset == "mosei_senti":
        return eval_mosei_senti(results, truths, True, wandb_logging=wandb_logging)
    elif hyp_params.dataset == 'mosi':
        return eval_mosi(results, truths, True, wandb_logging=wandb_logging)
    elif hyp_params.dataset == 'iemocap':
        return eval_iemocap(results, truths, wandb_logging=wandb_logging)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']        
    scheduler = settings['scheduler']
    
    print(model)

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, _) in enumerate(train_loader):
            _, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            preds, hiddens = model(text, audio, vision)
            if hyp_params.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += raw_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for _, (batch_X, batch_Y, _) in enumerate(loader):
                _, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                preds, _ = model(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 0
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        val_loss, val_results, val_truths = evaluate(model, criterion, test=False)
        test_loss, test_results, test_truths = evaluate(model, criterion, test=True)
        val_acc, val_f1 = get_eval_metrics(hyp_params, val_results, val_truths, wandb_logging=False)
        test_acc, test_f1 = get_eval_metrics(hyp_params, test_results, test_truths, wandb_logging=False)
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        if hyp_params.wandb: 
            wandb.log({"train_loss": train_loss, 
                "val_loss": val_loss, "test_loss": test_loss,
                "val_acc": val_acc, "test_acc": test_acc,
                "val_f1": val_f1, "test_f1": test_f1,
            }, step=epoch)

        print("-"*50)
        print(f'Epoch {epoch:2d} | Time {duration:5.4f} sec | Valid Loss {val_loss:5.4f} | Test Loss {test_loss:5.4f}')
        print(f'Valid Acc {val_acc:5.4f} | Test Acc {test_acc:5.4f} | Valid F1 {val_f1:5.4f} | Test F1 {test_f1:5.4f}')
        print("-"*50)
        
        if val_acc > best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_acc

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, criterion, test=True)

    print(f"Best accuracy and F1 {get_eval_metrics(hyp_params, results, truths, wandb_logging=hyp_params.wandb)}")

