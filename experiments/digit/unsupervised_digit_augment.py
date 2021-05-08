import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from data_load_aug import mnist
import matplotlib.pyplot as plt

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def digit_load(args): 
    train_bs = args.batch_size
    if args.dset == 'm':
        train_source = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]),
                strong_transform=transforms.Compose([
                    transforms.RandomResizedCrop(size=28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

    dset_loaders = {}
    dset_loaders["train"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_source, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    return dset_loaders

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            # inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent


# def compute_aug_loss(strong_inputs, target, netC, netB, netF):

#     k = strong_inputs.size(1)
#     strong_inputs = torch.cat([strong_inputs[:, i] for i in range(strong_inputs.size(1))], dim=0)

#     loss = torch.tensor(0.0)
#     outputs = netC(netB(netF(strong_inputs)))
#     for i in range(k):
#         # outputs = netC(netB(netF(strong_inputs[i])))
#         softmax_out = nn.Softmax(dim=1)(outputs[i*32:(i+1) * 32])
#         msoftmax = softmax_out.mean(dim=0)
#         loss += torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

#     return loss / k

def compute_aug_loss(strong_inputs, target, netC, netB, netF):

    loss = torch.tensor(0.0)
    for i in range(target.size(0)):
        outputs = netC(netB(netF(strong_inputs[i])))
        softmax_out = nn.Softmax(dim=1)(outputs)
        msoftmax = softmax_out.mean(dim=0)
        loss += torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

    return loss / target.size(0)


def train(args):

    ent_loss_record = []
    gent_loss_record = []
    sent_loss_record = []
    total_loss_record = []

    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u':
        netF = network.LeNetBase()#.cuda()
    elif args.dset == 'm':
        netF = network.LeNetBase()#.cuda()  
    elif args.dset == 's':
        netF = network.DTNBase()#.cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)#.cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)#.cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["train"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, strong_inputs, target = iter_source.next()
        except:
            iter_source = iter(dset_loaders["train"])
            inputs_source, strong_inputs, target = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source = inputs_source#.cuda()
        outputs_source = netC(netB(netF(inputs_source)))

        total_loss = torch.tensor(0.0)#.cuda()
        softmax_out = nn.Softmax(dim=1)(outputs_source)
        if args.ent:
            ent_loss = torch.mean(loss.Entropy(softmax_out))
            total_loss += ent_loss
            ent_loss_record.append(ent_loss.detach().cpu())

        if args.gent:
            msoftmax = softmax_out.mean(dim=0)
            gent_loss = -torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            gent_loss_record.append(gent_loss.detach().cpu())
            total_loss += gent_loss

        if args.sent:
            sent_loss = compute_aug_loss(strong_inputs, target, netC, netB, netF)
            total_loss += sent_loss
            sent_loss_record.append(sent_loss.detach().cpu())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        total_loss_record.append(total_loss.detach().cpu())

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print(iter_num, interval_iter, max_iter)
        #     netF.eval()
        #     netB.eval()
        #     netC.eval()
        #     acc_s_tr, _ = cal_acc(dset_loaders['train'], netF, netB, netC)
        #     acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
        #     log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te)
        #     args.out_file.write(log_str + '\n')
        #     args.out_file.flush()
        #     print(log_str+'\n')

        #     if acc_s_te >= acc_init:
        #         acc_init = acc_s_te
        #         best_netF = netF.state_dict()
        #         best_netB = netB.state_dict()
        #         best_netC = netC.state_dict()
            
        #     netF.train()
        #     netB.train()
        #     netC.train()

    best_netF = netF.state_dict()
    best_netB = netB.state_dict()
    best_netC = netC.state_dict()

    torch.save(best_netF, osp.join(args.output_dir, "F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "C.pt"))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(16,8))
    ax1.plot(list(range(len(ent_loss_record))), ent_loss_record, 'r')
    ax2.plot(list(range(len(gent_loss_record))), gent_loss_record, 'g')
    ax3.plot(list(range(len(sent_loss_record))), sent_loss_record, 'b')
    ax4.plot(list(range(len(total_loss_record))), total_loss_record, 'm')
    plt.tight_layout()
    plt.savefig(args.output_dir + '/loss.png')

    return netF, netB, netC

def test(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u':
        netF = network.LeNetBase()#.cuda()
    elif args.dset == 'm':
        netF = network.LeNetBase()#.cuda()  
    elif args.dset == 's':
        netF = network.DTNBase()#.cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)#.cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)#.cuda()

    args.modelpath = args.output_dir + '/F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = 'Task: {}, [DONT CARE] Accuracy = {:.2f}%'.format(args.dset, acc)
    try: 
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
    except:
        pass
    print(log_str+'\n')

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='s', choices=['u', 'm','s'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--gent', action='store_true', default=False)
    parser.add_argument('--ent', action='store_true', default=False)
    parser.add_argument('--sent', action='store_true', default=False)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()
    args.class_num = 10

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train(args)
    else:
        print('Já tem treino.. deletar!')
        # exit()
    
    test(args)

# python unsupervised_digit.py --dset m --gpu_id 0 --ent --output ckps_unsupervised_digit_ent
# python unsupervised_digit.py --dset m --gpu_id 0 --gent --output ckps_unsupervised_digit_gent
# python unsupervised_digit.py --dset m --gpu_id 0 --ent --gent --output ckps_unsupervised_digit_ent_gent

# na verdade n sem como saber qual classe vai sair .. ideal é ver tsne? ou mostrar as classificacoes primeiro?
# show classification + gradcam (versao mais rapida)