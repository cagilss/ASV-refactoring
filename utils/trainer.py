import lib.torch_ard as nn_ard
from config import cfg
import numpy as np
import torch
import torch.nn.functional as F
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def train(model, criterion, optimizer, tr_dataloader, epoch):
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, 
    #                        min_lr=1e-8, factor=0.001, verbose=True, eps=1e-8)
    loss, accuracy, total_num = 0, 0, 0

    model.train()
    for i_batch, sample_batched in enumerate(tr_dataloader):
        sample_id, inp_data, out_data, seq_len = sample_batched

        tensor_x = Variable(torch.FloatTensor(inp_data.float()), requires_grad=True).to(device) 
        tensor_y = Variable(torch.LongTensor(out_data), requires_grad=False).to(device)
        
        logits, _, _, _, _, alpha = model(tensor_x)
        
        if cfg.TRAIN.LOSS == 'EDL':
            annealing_step = 10
            global_step = epoch
            one_hot_label = one_hot_encoding(tensor_y)

            loss_batch = criterion(one_hot_label.to(device), alpha, global_step, annealing_step)
            loss_batch = torch.mean(loss_batch, dim=0)

        if cfg.TRAIN.LOSS == 'CrossEntropy':
            loss_batch = criterion(logits, tensor_y.view(-1))
        
        if cfg.TRAIN.LOSS == 'ELBO':
            loss_batch = nn_ard.ELBOLoss(model, F.cross_entropy)
            loss_batch = loss_batch.forward(logits, tensor_y.view(-1))
        
        # compute loss and accuracy
        _, pred_ = logits.max(dim=1)
        accuracy += (pred_.cpu() == tensor_y.view(-1).cpu().data).sum(dim=0)
        loss += loss_batch.item()
        total += inp_data.shape[0]

        optimizer.zero_grad()        
        loss_batch.backward()
        optimizer.step()

        del loss_batch, logits, alpha, tensor_x, tensor_y
    
    write_summary(epoch, loss, total, accuracy, model)
    write_summary(
        validate(model, criterion, optimizer, epoch, val_dataloader)
    )
    write_summary(
        validate(model, criterion, optimizer, epoch, eval_dataloader)
    )

    # save last model and pickle all the scores  

def write_summary(epoch, loss, total, accuracy, model):
    print('train dataset')
    print('EPOCH: {} Loss: {} Accuracy: {}'.format(
        epoch, (loss / total), (accuracy / total)))
    print('Sparsification ratio: %.3f%%' % (100.*nn_ard.get_dropped_params_ratio(model)))
    

def validate_(model, loss_func, optimizer, cuda, e, data, args):
    # test_(model, loss_func, optimizer, g, data[0], cuda, e, mode='train',args=args)
    # test_(model, loss_func, optimizer, g, data[1], cuda, e, mode='dev', args=args)
    # test_(model, loss_func, optimizer, g, data[2], cuda, e, mode='eval', args=args)

    # test_(model, loss_func, optimizer, data[0], cuda, e, mode='train',args=args)
    test_(model, loss_func, optimizer, data[1], cuda, e, mode='eval', args=args)


def validate(model, loss_func, optimizer, g, data, cuda, e, mode, args, roc_curve=False, 
plot_eer=False, save_scores=True):
    L_logits, L_softmax, L_probs, L_true_labels, L_sigmoid = [], [], [], [], []
    true_negatives, false_positives, false_negatives, true_positives, u, total, loss, acc \
        = 0, 0, 0, 0, 0, 0, 0, 0

    # paths
    logits_score_path = 'scores/{}_logit_scores_{}'.format(mode, e)
    sigmoid_scores_path = 'scores/{}_sigmoid_scores_{}'.format(mode, e)
    softmax_logits_score_path = 'scores/{}_softmax_scores_{}'.format(mode, e)
    probs_logits_score_path = 'scores/{}_probs_scores_{}'.format(mode, e)
    true_score_path = 'scores/{}_labels_{}'.format(mode, e)
    uncertainty_score_path = 'scores/{}_uncertainty_{}'.format(mode, e)

    model.eval()

    for i_batch, sample_batched in enumerate(data):
        sample_id, inp_data, out_data, seq_len = sample_batched

        tensor_x = Variable(torch.FloatTensor(inp_data.float()), requires_grad=False).to(device)
        tensor_y = Variable(torch.LongTensor(out_data), requires_grad=False).to(device)
        
        batch_size = tensor_x.size(0)
        
        # model output        
        logits, logsoftmax, softmax, uncertanity, prob, alpha = model(tensor_x)

        # calculate scores
        logits_np     = logits.data.cpu().numpy()
        logsoftmax_np = logsoftmax.data.cpu().numpy() 
        prob_np       = prob.data.cpu().numpy()
        softmax_np    = softmax.data.cpu().numpy()
        scores        = logits[:, 1] - logits[:, 0] # logit scores
        scores_sigmoid = torch.sigmoid(scores).data.cpu().numpy()
        true_labels   = tensor_y.data.cpu().numpy()

        L_logits.append(logits_np)
        L_sigmoid.append(scores_sigmoid)
        L_softmax.append(softmax_np)
        L_probs.append(prob_np)
        L_true_labels.append(true_labels)

        ##### EDL Loss Function #####
        #annealing_step = 10
        #global_step = e
        #one_hot_label = one_hot_encoding(tensor_y)
        #loss_batch = loss_func(one_hot_label.to(device), alpha, global_step, annealing_step)
        #loss_batch = torch.mean(loss_batch, dim=0)
        #loss += (loss_batch.item() * batch_size)

        ##### Cross Entropy Loss Function #####
        #tensor_y_cast = tensor_y.type_as(logits).reshape(-1,1) # for binary cross entropy 
        #loss_batch = loss_func(logits, tensor_y.view(-1))

        ##### ELBOLoss Function #####
        if cfg.TRAIN.LOSS == 'ELBO':
            loss_batch = nn_ard.ELBOLoss(model, F.cross_entropy)
            loss_batch = loss_batch.forward(logits, tensor_y.view(-1))

        loss += loss_batch.item()

        # 2D output
        _, pred_ = logits.max(dim=1)
        acc += (pred_.cpu() == tensor_y.view(-1).cpu().data).sum(dim=0)

        # compute uncertanity
        uncertanity_scalar = np.mean(uncertanity.data.cpu().numpy())
        u += uncertanity_scalar

        # confusion matrix
        pred_numpy = pred_.cpu().numpy() 
        try:
            tn, fp, fn, tp = confusion_matrix(true_labels, pred_numpy).ravel()
        except ValueError:
            pass
        true_negatives  += tn
        false_positives += fp
        false_negatives += fn
        true_positives  += tp 

        total += batch_size

        del loss_batch
        del logits
                    
    t_batch = int(np.round(total / batch_size))
    u      /= t_batch
    acc_    = acc.item() / total
    loss   /= total


    if save_scores:
        np.save(logits_score_path, L_logits)
        np.save(sigmoid_scores_path, L_sigmoid)
        np.save(softmax_logits_score_path, L_softmax)
        np.save(probs_logits_score_path, L_probs)
        np.save(true_score_path, L_true_labels)
        np.save(uncertainty_score_path, u)
         
    print('epoch: {},  {} -> loss:{}, acc:{}, uncer: {}'    
    .format(e, mode, loss, acc_, u))
    print('true_negatives: {}, true_positives: {}, false_negatives: {}, false_positives: {}'
    .format(true_negatives, true_positives, false_negatives, false_positives))
    print('-------')
    print('Sparsification ratio: %.3f%%' % (100.*nn_ard.get_dropped_params_ratio(model)))
    if mode == 'eval': print('--------------------------------------------------------------', os.linesep)

    return {'acc': acc, 'loss': loss}


def one_hot_encoding(label):
    List = torch.LongTensor(np.zeros((label.shape[0], 2)))
    for idx,L in enumerate(label):
        List[idx][L] = 1
    return List