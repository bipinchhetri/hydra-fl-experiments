import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random


from model import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                #new
                # net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
                # net = SimpleCNN_shallow(16*5*5, 120)
                net = SimpleCNN_MNIST_shallow(16*5*5, 120)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def compute_attack_indicator(user_updates, global_model, train_dl_global, test_dl, args, device, prev_test_acc=None):
    """
    Compute attack indicator A_t based on gradient divergence and model performance.
    Returns A_t in [0, 1], where 0 is benign, 1 is attack.
    """
    # Gradient divergence (cosine similarity variance)
    cos = torch.nn.CosineSimilarity(dim=-1)
    similarities = []
    for i in range(len(user_updates)):
        for j in range(i + 1, len(user_updates)):
            sim = cos(user_updates[i], user_updates[j]).item()
            similarities.append(sim)
    sim_variance = np.var(similarities) if similarities else 0.0

    # Model performance drop
    test_acc, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, return_projections = False, device="cpu", multiloader=False)
    #test_acc, _ = compute_accuracy(global_model, test_dl, args, get_confusion_matrix=False, device=device)
    print("Checking test acc from attack indicator function:", test_acc)
    
    perf_drop = 0.0
    if prev_test_acc is not None:
        perf_drop = max(0, prev_test_acc - test_acc)
    
    # Combine metrics (example weights, tune as needed)
    sim_threshold = 0.1  # Tune based on dataset
    perf_threshold = 0.1  # Tune based on dataset
    sim_score = min(sim_variance / sim_threshold, 1.0)
    perf_score = min(perf_drop / perf_threshold, 1.0)
    A_t = 0.5 * sim_score + 0.5 * perf_score  # Weighted combination
    print("A_t from attack indicator:", A_t, "test_acc from attack indicator:", test_acc)
    return A_t, test_acc


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc,_ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix,_ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:
            train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    print('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    print('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    net = nn.DataParallel(net)
    net.cuda()
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


# def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
#                       round, device="cpu", attack_indicator=0.2):
    
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # print(device)
#     # net.cuda()
#     # #net = nn.DataParallel(net)
#     # net = torch.nn.DataParallel(net, device_ids=[0])  # Then wrap with DataParallel

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     net = net.to(device)
#     net = torch.nn.DataParallel(net, device_ids=[0])

#     global_net = global_net.to(device)
#     global_net = torch.nn.DataParallel(global_net, device_ids=[0])

#     previous_nets = [torch.nn.DataParallel(prev_net.to(device), device_ids=[0]) for prev_net in previous_nets]


#     #net.to(device)
#     logger.info('Training network %s' % str(net_id))
#     logger.info('n_training: %d' % len(train_dataloader))
#     logger.info('n_test: %d' % len(test_dataloader))

#     train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

#     test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

#     logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
#     logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


#     if args_optimizer == 'adam':
#         optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
#     elif args_optimizer == 'amsgrad':
#         optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
#                                amsgrad=True)
#     elif args_optimizer == 'sgd':
#         optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
#                               weight_decay=args.reg)

#     criterion = nn.CrossEntropyLoss().cuda()
#     # global_net.to(device)

#     # for previous_net in previous_nets:
#     #     previous_net.cuda()
#     global_w = global_net.state_dict()

#     cnt = 0
#     cos=torch.nn.CosineSimilarity(dim=-1)
#     # mu = 0.001

#     beta_max, beta_min = 1.0, 0.25
#     gamma_max, gamma_min = 2.0, 1.0
#     b_max, b_min = 4.0, 1.0
#     alpha = 0.2  # Smoothing factor
#     beta_t = beta_max
#     gamma_t = gamma_min
#     b_t = b_min

#     # Optional: Client-side discrepancy for hybrid detection
#     kl_div = compute_model_discrepancy(net, global_net, train_dataloader, device)
#     A_t = 0.2  # Use server-provided A_t; optionally adjust with kl_div
#     if kl_div > 0.3:  # Tune threshold
#         A_t = max(A_t + 0.2, 1.0)  # Boost A_t if high discrepancy


#     for epoch in range(epochs):
#         epoch_loss_collector = []
#         epoch_loss1_collector = []
#         epoch_loss2_collector = []
#         epoch_loss3_collector = []  # For shallow KD loss


#         # Update dynamic coefficients
#         beta_t = alpha * (beta_max * (1 - A_t) + beta_min * A_t) + (1 - alpha) * beta_t
#         #print("beta_t:", beta_t)
#         gamma_t = alpha * (gamma_min * (1 - A_t) + gamma_max * A_t) + (1 - alpha) * gamma_t
#         #print("gamma_t:", gamma_t)
#         b_t = alpha * (b_min * (1 - A_t) + b_max * A_t) + (1 - alpha) * b_t
#         #print("b_t:", b_t)
#         for batch_idx, (x, target) in enumerate(train_dataloader):
#             #x, target = x.cuda(), target.cuda()
#             x, target = x.to(device), target.to(device)

#             optimizer.zero_grad()
#             x.requires_grad = False
#             target.requires_grad = False
#             target = target.long()


#             #new
#             sr1, so1, sr2, so2, pro1, out = net(x)
#             _, _, _, _, pro2, _ = global_net(x)
#             # _, pro1, out = net(x)
#             # _, pro2, _ = global_net(x)

#             # #new
#             # if(net_id<2):
#             #     pro2 *= -1

#             #new
#             posi = cos(sr2, pro2)
#             logits = posi.reshape(-1,1)

#             #new
#             for previous_net in previous_nets:
#                 previous_net.cuda()
#                 _, _, _, _, pro3, _ = previous_net(x)
#                 nega = cos(pro1, pro3)
#                 logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

#                 previous_net.to('cpu')

#             logits /= temperature
#             labels = torch.zeros(x.size(0)).cuda().long()

#             loss2 = mu * criterion(logits, labels)


#             loss1 = criterion(out, target)

#             # Shallow KD loss (use second shallow block)
#             shallow_kd_loss = criterion(so2, target)

#             #new
#             #loss = loss1 + 0.1*loss2
#             # loss = loss1
#             loss = loss1 + (beta_t / b_t) * loss2 + gamma_t * shallow_kd_loss

#             loss.backward()
#             optimizer.step()

#             cnt += 1
#             epoch_loss_collector.append(loss.item())
#             epoch_loss1_collector.append(loss1.item())
#             epoch_loss2_collector.append(loss2.item())
#             epoch_loss3_collector.append(shallow_kd_loss.item())

#         epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
#         epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
#         epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
#         epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)
#         #logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))
#         logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f Loss3: %f Beta: %f Gamma: %f b: %f A_t: %f' % 
#             (epoch, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3, beta_t, gamma_t, b_t, A_t))


#     # for previous_net in previous_nets:
#     #     previous_net.to('cpu')
#     for previous_net in previous_nets:
#         previous_net.to('cpu')
#     net.to('cpu')
#     train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
#     #new
#     test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
#     # test_acc, _, proj_list, true_labels_list = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, return_projections=True, device=device)
#     # torch.save(proj_list, 'proj_list_moon2_attack'+str(net_id))
#     # torch.save(true_labels_list, 'true_labels_list_moon2_attack'+str(net_id))

#     logger.info('>> Training accuracy: %f' % train_acc)
#     logger.info('>> Test accuracy: %f' % test_acc)
#     net.to('cpu')
#     logger.info(' ** Training complete **')
#     return train_acc, test_acc



def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu", attack_indicator=0.2):

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    # Move models to device before wrapping in DataParallel
    net = net.to(device)
    #net = torch.nn.DataParallel(net, device_ids=[0])

    global_net = global_net.to(device)
    #global_net = torch.nn.DataParallel(global_net, device_ids=[0])

    previous_nets = [torch.nn.DataParallel(pn.to(device), device_ids=[0]) for pn in previous_nets]

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # print('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # print('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # Optimizer
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    global_w = global_net.state_dict()

    cos = torch.nn.CosineSimilarity(dim=-1)
    beta_max, beta_min = 1.0, 0.25
    gamma_max, gamma_min = 2.0, 1.0
    b_max, b_min = 4.0, 1.0
    alpha = 0.2
    beta_t = beta_max
    gamma_t = gamma_min
    b_t = b_min

    # Optional: Client-side discrepancy for hybrid detection
    kl_div = compute_model_discrepancy(net, global_net, train_dataloader, device)
    A_t = 0.2
    if kl_div > 0.3:
        A_t = max(A_t + 0.2, 1.0)

    # print("A_t:", A_t)
    # print("kl_div:", kl_div)
    for epoch in range(epochs):
        epoch_loss_collector, epoch_loss1_collector, epoch_loss2_collector, epoch_loss3_collector = [], [], [], []

        beta_t = alpha * (beta_max * (1 - A_t) + beta_min * A_t) + (1 - alpha) * beta_t
        gamma_t = alpha * (gamma_min * (1 - A_t) + gamma_max * A_t) + (1 - alpha) * gamma_t
        b_t = alpha * (b_min * (1 - A_t) + b_max * A_t) + (1 - alpha) * b_t
        # print("beta_t:", beta_t)
        # print("gamma_t:", gamma_t)
        # print("b_t:", b_t)
        for batch_idx, (x, target) in enumerate(train_dataloader):
            net.to(device)
            global_net.to(device)
            x, target = x.to(device), target.to(device)
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            optimizer.zero_grad()

            sr1, so1, sr2, so2, pro1, out = net(x)
            _, _, _, _, pro2, _ = global_net(x)

            posi = cos(sr2, pro2)
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                _, _, _, _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            logits /= temperature
            labels = torch.zeros(x.size(0)).long().to(device)

            loss1 = criterion(out, target)
            loss2 = mu * criterion(logits, labels)
            shallow_kd_loss = criterion(so2, target)

            loss = loss1 + (beta_t / b_t) * loss2 + gamma_t * shallow_kd_loss

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
            epoch_loss3_collector.append(shallow_kd_loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)

        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f Loss3: %f Beta: %f Gamma: %f b: %f A_t: %f' %
                    (epoch, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3, beta_t, gamma_t, b_t, A_t))

    # Move previous nets back to CPU to save memory
    for previous_net in previous_nets:
        previous_net.to('cpu')

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc



def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu", attack_indicator=0.0):
    avg_acc = 0.0
    acc_list = []
    acc_list_local = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if args.alg == 'fedavg':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device, attack_indicator=attack_indicator)

        elif args.alg == 'local_training':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        test_acc_local, _ = compute_accuracy(global_model, test_dl_local, device=device)
        avg_acc += testacc
        acc_list.append(testacc)
        acc_list_local.append(test_acc_local)
    avg_acc /= args.n_parties
    print("avg test acc %f" % avg_acc)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    logger.info("per-client accuracies: %s" % str(acc_list_local))
    return nets, avg_acc

def full_trim(v, f):
        '''
        Full-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised.
        v: the list of squeezed gradients
        f: the number of compromised worker devices
        '''
        vi_shape = v[0].unsqueeze(0).T.shape
        v_tran = v.T

        maximum_dim = torch.max(v_tran, dim=1)
        maximum_dim = maximum_dim[0].reshape(vi_shape)
        minimum_dim = torch.min(v_tran, dim=1)
        minimum_dim = minimum_dim[0].reshape(vi_shape)
        direction = torch.sign(torch.sum(v_tran, dim=-1, keepdims=True))
        directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

        for i in range(f):
            random_12 = 2
            #         random_12 = random.randint(0,9)
            tmp = directed_dim * (
                        (direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
            tmp = tmp.squeeze()
            # tmp = torch.FloatTensor(tmp.shape).uniform_(-1, 1)
            # tmp = torch.FloatTensor(tmp.shape).uniform_(minimum_dim, maximum_dim)
            v[i] = tmp
            # v[i] = 1 * torch.rand_like(tmp)
            # v[i] = tmp*0
        return v


def tr_mean(all_updates, n_attackers):
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates, 0)
    return out

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    #new
    experiment = "moon. mnist. trim attack. 2/10. 10 local e. beta 0.05, shallow distillation 2. mu=0.1"
    # experiment = "moon. cifar10. trim attack. 2/10. 10 local e. beta 0.5, shallow distillation 2."
    # experiment = "moon. trim attack. 2/10. 10 local e. beta 5, mu=10, shallow distillation 2."
    # experiment = "inverting pro2 in local_train_fedcon. trimmed mean. 2/10 malicious clients. 10 local epochs. beta 0.1"
    # experiment = "fedavg baseline. trim attack. trimmed mean. 2/10 malicious clients. 10 local epoch. beta 5"
    logger.info(experiment)

    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cuda')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cuda')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cuda')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        max_acc = 0
        prev_test_acc = None
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            # user_updates = []
            # for net_id, net in enumerate(nets_this_round.values()):
            #     params = []
            #     for _, (name, param) in enumerate(net.state_dict().items()):
            #         params = param.view(-1).data if len(params) == 0 else torch.cat((params, param.view(-1).data))
            #     user_updates = params[None, :] if len(user_updates) == 0 else torch.cat((user_updates, params[None, :]), 0)

            # print(user_updates.shape)

            # #new
            # user_updates = full_trim(user_updates, 2)

            # # agg_updates = torch.mean(user_updates, dim = 0)
            # agg_updates = tr_mean(user_updates, 2)

            A_t, curr_test_acc = compute_attack_indicator(user_updates, global_model, train_dl_global, test_dl, args, device, prev_test_acc)
            # prev_test_acc = curr_test_acc
            # logger.info(f"Attack Indicator A_t: {A_t:.4f}")
            nets, avg_acc = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device, attack_indicator=0.2)



            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            # for net_id, net in enumerate(nets_this_round.values()):
            #     net_para = net.state_dict()
            #     if net_id == 0:
            #         for key in net_para:
            #             global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            #     else:
            #         for key in net_para:
            #             global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            user_updates = []
            for net_id, net in enumerate(nets_this_round.values()):
                params = []
                for _, (name, param) in enumerate(net.state_dict().items()):
                    params = param.view(-1).data if len(params) == 0 else torch.cat((params, param.view(-1).data))
                user_updates = params[None, :] if len(user_updates) == 0 else torch.cat((user_updates, params[None, :]), 0)

            print(user_updates.shape)

            #new
            user_updates = full_trim(user_updates, 2)

            # agg_updates = torch.mean(user_updates, dim = 0)
            agg_updates = tr_mean(user_updates, 2)

            print(np.shape(agg_updates))


            start_idx = 0
            state_dict = {}
            previous_name = 'none'
            for i, (name, param) in enumerate(global_model.state_dict().items()):
                start_idx = 0 if i == 0 else start_idx + len(
                    global_model.state_dict()[previous_name].data.view(-1))
                start_end = start_idx + len(global_model.state_dict()[name].data.view(-1))
                #                     print(np.shape(user_updates[k][start_idx:start_end]))
                params = agg_updates[start_idx:start_end].reshape(
                    global_model.state_dict()[name].data.shape)
                #                     params = local_models[0][0].state_dict()[name]
                state_dict[name] = params
                previous_name = name
            
            global_model.load_state_dict(state_dict)

            # if args.server_momentum:
            #     delta_w = copy.deepcopy(global_w)
            #     for key in delta_w:
            #         delta_w[key] = old_w[key] - global_w[key]
            #         moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
            #         global_w[key] = old_w[key] - moment_v[key]

            # global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            #new
            # test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, device=device)
            test_acc, _, proj_list, true_labels_list = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, return_projections=True, device=device)
            # torch.save(proj_list, 'proj_list_moon2_attack')
            # torch.save(true_labels_list, 'true_labels_list_moon2_attack')
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            # print("Global Model Test accuracy: ", test_acc)
            # print("Global Model Train accuracy: ", train_acc)
            print("avg acc: ", avg_acc)
            if (test_acc < avg_acc):
                max_acc = avg_acc
            print("Round: ", round, " Max Accuracy: ", max_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)


            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')


    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            user_updates = []
            for net_id, net in enumerate(nets_this_round.values()):
                params = []
                for _, (name, param) in enumerate(net.state_dict().items()):
                    params = param.view(-1).data if len(params) == 0 else torch.cat((params, param.view(-1).data))
                user_updates = params[None, :] if len(user_updates) == 0 else torch.cat((user_updates, params[None, :]), 0)

            print(user_updates.shape)

            # user_updates = full_trim(user_updates, 2)

            # agg_updates = torch.mean(user_updates, dim = 0)
            agg_updates = tr_mean(user_updates, 2)


            start_idx = 0
            state_dict = {}
            previous_name = 'none'
            for i, (name, param) in enumerate(global_model.state_dict().items()):
                start_idx = 0 if i == 0 else start_idx + len(
                    global_model.state_dict()[previous_name].data.view(-1))
                start_end = start_idx + len(global_model.state_dict()[name].data.view(-1))
                #                     print(np.shape(user_updates[k][start_idx:start_end]))
                params = agg_updates[start_idx:start_end].reshape(
                    global_model.state_dict()[name].data.shape)
                #                     params = local_models[0][0].state_dict()[name]
                state_dict[name] = params
                previous_name = name
            
            global_model.load_state_dict(state_dict)

            # for net_id, net in enumerate(nets_this_round.values()):
            #     net_para = net.state_dict()
            #     if net_id == 0:
            #         for key in net_para:
            #             global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            #     else:
            #         for key in net_para:
            #             global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            # if args.server_momentum:
            #     delta_w = copy.deepcopy(global_w)
            #     for key in delta_w:
            #         delta_w[key] = old_w[key] - global_w[key]
            #         moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
            #         global_w[key] = old_w[key] - moment_v[key]


            # global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')

