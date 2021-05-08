import torch
import torch.optim as optim
import torch.nn as nn
import numpy.matlib as matlib
import numpy as np
import matplotlib.pyplot as plt

def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    entropy = torch.mean(entropy)
    return entropy 

def Diversity(input_):
    K = input_.size(1)
    bs = input_.size(0)

    input_ = input_.mean(dim=0)
    diversity = -input_ * torch.log(input_ + 1e-5)
    diversity = -torch.sum(diversity, dim=0)
    
    v = np.array([bs / K] * K) / bs
    v = -v * np.log(v + 1e-5)
    v = -np.sum(v)
    # print('min: ', v)
    return diversity, v

def format_tensor(tensor):

    fmt = []
    for i in range(tensor.size(0)):
        fmt.append([])
        for j in range(tensor.size(1)):
            fmt[i].append('{:.4f}'.format(tensor[i, j].item()))
    return fmt


def train():

    # batch = [[0.03, 0.011, 0.012],
    #          [0.011, 0.03, 0.012],
    #          [0.011, 0.012, 0.03]]
    # batch = [[6.0, 0.4, 3.0],
    #          [0.2, 3.0, 1.2],
    #          [0.1, 1.0, 2.0]]
    # batch = [[0.0, 0.0, 0.0],
    #          [0.0, 0.0, 0.0],
    #          [0.0, 0.0, 0.0]]
    batch = [[1.3, 1.1, 1.1],
             [1.3, 1.1, 1.1],
             [1.3, 1.1, 1.1]]
    batch = nn.Parameter(torch.tensor(batch, requires_grad=True))

    label = [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]]
    label = torch.tensor(label)

    ent_loss_record = []

    optimizer = optim.SGD([batch], lr=0.01, momentum=0.9)

    for _ in range(5000):

        softmax_out = nn.Softmax(dim=1)(batch)

        # loss = torch.nn.functional.l1_loss(softmax_out, label) 
        # loss = Entropy(softmax_out) 
        loss, min_div = Diversity(softmax_out)
        # loss = Entropy(softmax_out) + Diversity(softmax_out)

        # print('Loss: {:.4f} | Softmax: {} | Values: {}'.format(loss.item(), ['{:.4f}'.format(x) for x in softmax_out.tolist()], ['{:.4f}'.format(x) for x in batch.tolist()]))
        print('Loss: {:.4f} / {:.4f} | Softmax: {} | Values: {}'.format(loss.item(), min_div, format_tensor(softmax_out), format_tensor(batch)))
        # print('Loss: {:.4f} | Softmax: {} | Values: {}'.format(loss.item(), softmax_out.tolist(), batch.tolist()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # print('Loss: {:.4f} | Softmax: {} | Values: {}'.format(loss.item(), ['{:.4f}'.format(x) for x in softmax_out.tolist()], ['{:.4f}'.format(x) for x in batch.tolist()]))
    # print('Loss: {:.4f} | Softmax: {} | Values: {}'.format(loss.item(), softmax_out.tolist(), batch.tolist()))
    print('Loss: {:.4f} / {:.4f} | Softmax: {} | Values: {}'.format(loss.item(), min_div, format_tensor(softmax_out), format_tensor(batch)))


    # acc_init = 0
    # max_iter = args.max_epoch * len(dset_loaders["train"])
    # interval_iter = max_iter // 10
    # iter_num = 0

    # netF.train()
    # netB.train()
    # netC.train()

    # while iter_num < max_iter:
    #     try:
    #         inputs_source, _ = iter_source.next()
    #     except:
    #         iter_source = iter(dset_loaders["train"])
    #         inputs_source, _ = iter_source.next()

    #     if inputs_source.size(0) == 1:
    #         continue

    #     iter_num += 1
    #     lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

    #     inputs_source = inputs_source.cuda()
    #     outputs_source = netC(netB(netF(inputs_source)))

    #     total_loss = torch.tensor(0.0).cuda()
    #     softmax_out = nn.Softmax(dim=1)(outputs_source)
    #     if args.ent:
    #         ent_loss = torch.mean(loss.Entropy(softmax_out))
    #         total_loss += ent_loss
    #         ent_loss_record.append(ent_loss.detach().cpu())

    #     if args.gent:
    #         msoftmax = softmax_out.mean(dim=0)
    #         gent_loss = -torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    #         gent_loss_record.append(gent_loss.detach().cpu())
    #         total_loss += gent_loss
        
    #     optimizer.zero_grad()
    #     total_loss.backward()
    #     optimizer.step()
    #     total_loss_record.append(total_loss.detach().cpu())

    #     if iter_num % interval_iter == 0 or iter_num == max_iter:
    #         netF.eval()
    #         netB.eval()
    #         netC.eval()
    #         acc_s_tr, _ = cal_acc(dset_loaders['train'], netF, netB, netC)
    #         acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    #         log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}%/ {:.4f}%'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te)
    #         args.out_file.write(log_str + '\n')
    #         args.out_file.flush()
    #         print(log_str+'\n')

    #         if acc_s_te >= acc_init:
    #             acc_init = acc_s_te
    #             best_netF = netF.state_dict()
    #             best_netB = netB.state_dict()
    #             best_netC = netC.state_dict()
            
    #         netF.train()
    #         netB.train()
    #         netC.train()

    # torch.save(best_netF, osp.join(args.output_dir, "F.pt"))
    # torch.save(best_netB, osp.join(args.output_dir, "B.pt"))
    # torch.save(best_netC, osp.join(args.output_dir, "C.pt"))

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(16,8))
    # ax1.plot(list(range(len(ent_loss_record))), ent_loss_record, 'r')
    # ax2.plot(list(range(len(gent_loss_record))), gent_loss_record, 'g')
    # ax3.plot(list(range(len(total_loss_record))), total_loss_record, 'b')
    # plt.tight_layout()
    # plt.savefig(args.output_dir + '/loss.png')

    # return netF, netB, netC

train()