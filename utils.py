# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_noniid_unequal
import numpy as np


def newmodel(args, model):
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    #newmodel = copy.deepcopy(global_model)
    model.to(device)
    model.train()

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = cifar_noniid_unequal(train_dataset,args.num_users)
                print('i am non iid unequal')
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
        else:
            data_dir = 'data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                print('i am non iid unequal')
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups
def average_weights(w,s):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    m=sum(s)
    for key in w_avg.keys():
        w_avg[key] = s[0]*w_avg[key]
        for i in range(1, len(w)):
            w_avg[key] += s[i]*w[i][key]
        w_avg[key] = torch.div(w_avg[key], m)
    return w_avg
#'''
'''
entropy = []
for k, v in d.items():
    en=0
    counts=np.zeros(10)
    for i in v: 
        _,lbl = dataset_train[int(i)]
        counts[lbl]+=1

    for j in range (len(counts)):
        counts[j]=counts[j]/len(v)
        if(counts[j]!=0):
            en+= -counts[j]*math.log(counts[j])

    entropy.append(en)
print(entropy)
'''

def get_entropy(user_groups,dataset_train):
    entropy = []
    for k, v in user_groups.items():
        en=0
        counts=np.zeros(10)
        for i in v: 
            _,lbl = dataset_train[int(i)]
            counts[lbl]+=1

        for j in range (len(counts)):
            counts[j]=counts[j]/len(v)
            if(counts[j]!=0):
                en+= -counts[j]*math.log(counts[j])
        entropy.append(en)
    return entropy

def get_gini(user_groups,dataset_train):
    entropy = []
    #dico = {}
    for k, v in user_groups.items():
        
        en = 0
        counts = np.zeros(10)
        for i in v: 
            _,lbl = dataset_train[int(i)]
            counts[lbl] += 1

        for j in range (len(counts)):
            counts[j] = counts[j]/len(v)
            en += counts[j]**2
        en = 1 - en
        entropy.append(en)
        #dico[str(i)] = en
    return entropy


def flip_labels(user_groups,dataset_train, malicious_ids, target_label,malicious_label):
    dataset_train1 = []
    for k, v in user_groups.items():
        counts=np.zeros(10)
        
        if(k in malicious_ids):
            for i in v: 
                img,lbl = dataset_train[int(i)]
                if(lbl==target_label):
                    ds1 = list(dataset_train[int(i)])
                    ds = img, malicious_label
                    dataset_train1.append(tuple(ds))
                else:
                    dataset_train1.append(dataset_train[int(i)])
        else:
            for i in v: 
                dataset_train1.append(dataset_train[int(i)])
    return dataset_train1


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def get_importance(entropy,size,roE=1/2,roD=1/2):
    importance = []
    totalsize = sum(size)
    print('entropy',entropy)
    print('sizes',size)
    print(totalsize)
    for i in range(len(entropy)):
        importance.append(roE*entropy[i] + roD * size[i]/totalsize )
    return importance


def get_importance(entropy,size,age,rnd,roE=1/3,roD=1/3,roA =1/3):
    importance = []
    totalsize = sum(size)
    for i in range(len(entropy)):
        importance.append(roE*entropy[i] + roD * size[i]/totalsize + roA*age[i]/rnd )
    return importance


def get_order(importance):
    e=[]
    for i in range(len(importance)):
        e.append(-1*importance[i])
    return np.argsort(e)

def get_age(age):
    a = []
    for i in range(len(age)):
        a.append(math.log2(1+age[i]))
    return a

def get_reputation(local_accuracies, test_accuracies, idxs, reputation,b1=0.5,b2=0.5):
    avg = np.array(test_accuracies).mean()
    print('average acc',avg)
    k = 0
    for i in idxs:
        print('calculation reputation for',i)
        print(local_accuracies[k])
        print(test_accuracies[k])        
        reputation[i] = reputation[i] - (b1*(local_accuracies[k]-avg)+b2*(local_accuracies[k]-test_accuracies[k]))
        if reputation[i]>1:
            reputation[i] = 1
        if reputation[i]<0:
            reputation[i] = 0 
        k +=1    
    return reputation

def get_measure(diversity,reputation,w1=0.5,w2=0.5):
    #print([w1*diversity[i]+ w2*reputation[i] for i in range(len(diversity))])
    return [w1*diversity[i]+ w2*reputation[i] for i in range(len(diversity))]

def get_preference(i1,i2,i3):
    L= []
    M = []
    for c in range(len(i1)):
        L.append([i1[c],i2[c],i3[c]])
        e=[]
        for i in range(len(L)):
            e.append(-1*L[i])
        M.append(np.argmax(e))
    return M
def get_preference_test(i1,i2):
    L= []
    M = []
    for c in range(len(i1)):
        L.append([i1[c],i2[c]])
        e=[]
        for i in range(len(L[c])):
            e.append(-1*L[c][i])
        M.append(np.argmax(e))
    return M
    
if __name__=='__main__':
    from options import args_parser 

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset1, test_dataset1, user_groups1 = get_dataset(args)
    malicious_ids = [i for i in range(10)]
    #malicious_ids = [0]
    target_label,malicious_label = 6,2
    dataset_train1 = flip_labels(user_groups1 ,train_dataset1, malicious_ids, target_label,malicious_label)
    print(len(dataset_train1))
    for k, v in user_groups1.items():
        en=0
        counts=np.zeros(10)
        for i in v: 
            _,lbl = dataset_train1[int(i)]
            counts[lbl]+=1

    print(counts)
    # load dataset and user groups
    # train_dataset1, test_dataset1, user_groups1 = get_dataset(args)
    # args.dataset = 'fmnist'
    # train_dataset2, test_dataset2, user_groups2 = get_dataset(args)
    # args.dataset = 'cifar'
    # train_dataset3, test_dataset3, user_groups3 = get_dataset(args)
    # E1 = get_gini(user_groups1,train_dataset1)
    # S1 = [len(user_groups1[idx]) for idx in range(len(user_groups1))]
    # E2 = get_gini(user_groups2,train_dataset2)
    # S2 = [len(user_groups2[idx]) for idx in range(len(user_groups2))]
    # E3 = get_gini(user_groups3,train_dataset3)
    # S3 = [len(user_groups3[idx]) for idx in range(len(user_groups3))]
    # importance1 = get_importance(E1,S1)
    # importance2 = get_importance(E2,S2)
    # importance3 = get_importance(E3,S3)
    # M = get_preference(importance1,importance2,importance3)
    # M = get_preference_test(importance1,importance2)
    # print(M)

    # args.dataset = 'cifar'
    # train_dataset3, test_dataset3, user_groups3 = get_dataset(args)
    # print(user_groups3)
    #E3 = get_gini(user_groups3,train_dataset3)
    #S3 = [len(user_groups3[idx]) for idx in range(len(user_groups3))]

