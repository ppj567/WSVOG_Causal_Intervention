import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import copy
import random
import errno
import argparse
import torch
torch.set_num_threads(4)
CUDA = True if torch.cuda.is_available() else False

from torch.optim import lr_scheduler
from model_mm import WSTOG
from data_loader_mm import YC2_train_data, YC2_test_data, collate_data

from datetime import datetime
from tools.test_tool import test

from torch.utils.data import Dataset, DataLoader
from config import TfConfig
from tqdm import tqdm


def pause():
    programPause = input("Press the <ENTER> key to continue...")


if torch.cuda.is_available():
    print("CUDA is OK")

def main(args, paras):
    try:
        os.makedirs(args.checkpoint_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    print('loading dataset')

    train_set, val_set, test_set = get_dataset(args)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle = False, num_workers = 4)

    train_data_num = train_set.__len__()
    val_data_num = val_set.__len__()

    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, pin_memory=True)
    val_loader  = DataLoader(val_set, batch_size=1, shuffle = False, num_workers = 4)


    print('building model')
    model = WSTOG(mem_slots_num=args.mem_slots_num,
                  input_feature_dims=args.input_feature_dims,
                  input_label_dims=args.input_label_dims,
                  embedding_size=args.embedding_size,
                  drop_crob=args.drop_crob,
                  batch_size=args.batch_size,
                  max_num_obj=args.max_num_obj,
                  num_classes=args.num_classes,
                  dataset_feat=args.feature,
                  beta=args.beta,
                  args=args,
                  paras=paras)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= args.lr, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= args.step, gamma = 1e-1)



    best_accuracy = 0.
    test_reslut_list=[]
    'start training'

    if paras.train_test == 'test':
        pretrained_dict = torch.load(paras.model_dir)
        mymodel = load_my_dicts(model, pretrained_dict, args.model_name[paras.model_type])
        print("Evluation on validation set: ")
        test_result, macro_box_acc, micro_box_acc, macro_qry_acc, micro_qry_acc = test(mymodel, val_loader, args, paras)
        print("Evluation on test set: ")
        test_result, macro_box_acc, micro_box_acc, macro_qry_acc, micro_qry_acc = test(mymodel, test_loader, args, paras)
        exit()

    if paras.initialize:
        init_dict = torch.load(paras.initialize)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in init_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    for epoch in range(1, args.max_epochs + 1):
        epoch_loss = 0.
        train_loss = 0.
        flag=0
        output_str = "%s " %(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        model.cuda()
        model.train()

        for batch_cnt, batch in enumerate(tqdm(train_loader)):
            pos_feature, pos_label, neg_feature, neg_label = batch

            pos_feature = pos_feature.cuda()
            pos_label = pos_label.cuda()

            neg_feature = neg_feature.cuda()
            neg_label = neg_label.cuda()


            optimizer.zero_grad()

            cost = model(pos_label, pos_feature, neg_label, neg_feature, is_training=True, is_finetune=True)
            cost.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 10.0)
            optimizer.step()

            epoch_loss += cost.item()
            train_loss += cost.item()

            if (batch_cnt+1) % int(10) == 0:
                train_loss = 0.

        print(output_str, ' epoch: %d,  epoch_loss: %lf, lr: %lf, beta: %lf, batchsize: %lf, grad_norm: %lf' % (epoch, epoch_loss / float(len(train_loader)), optimizer.param_groups[0]['lr'], args.beta, args.batch_size, grad_norm))

        if epoch % 1 == 0:
            model_name = paras.train_mode + '_epoch{}.pkl'.format(epoch)
            print(model_name + ' saved!')
            torch.save(model.state_dict(),'./checkpoint/'+model_name)
            print("Evluation on validation set: ")
            test_result, macro_box_acc, micro_box_acc, macro_qry_acc, micro_qry_acc = test(model, val_loader, args, paras)
            current_accuracy = macro_box_acc+micro_box_acc+macro_qry_acc+micro_qry_acc
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                model_best = paras.train_mode + '_Best.pkl'
                torch.save(model.state_dict(),'./checkpoint/'+model_best)
            exp_lr_scheduler.step()

        if epoch % 1 == 0:
            print("Evluation on test set: ")

            test_result, macro_box_acc, micro_box_acc, macro_qry_acc, micro_qry_acc = test(model, test_loader, args, paras)
            test_reslut_list.append((epoch, test_result))


def load_my_dict(model, pretrained_dict, key=None):
    if isinstance(key,str):
        pretrained_dict = pretrained_dict[key]
    model = copy.deepcopy(model)
    model_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict_selected)
    model.load_state_dict(model_dict)
    return model

def load_my_dicts(model, pretrained_dict, keylist=None):
    mymodel = []
    if isinstance(keylist,list) and len(keylist)>1:
        for item in keylist:
            mymodel.append(load_my_dict(model, pretrained_dict, item))
    else:
        mymodel.append(load_my_dict(model, pretrained_dict))
    return mymodel

def get_dataset(args):
    if args.feature == "CVPR":
        print("using YC2 dataset from CVPR -4096")
        args.input_feature_dims=4096
        args.lr = 8e-4
        args.weight_decay = 1e-4
        args.step = 40
        args.batch_size =48
        args.beta = 0.2
        train_set = YC2_train_data(label_file=args.yc2_train_file,
                                   roi_pooled_feat_root=args.cvpr_rpn_feat_root,
                                   data_split='training',
                                   rpn_proposal_root=args.cvpr_proposal_root,
                                   num_proposals=20,
                                   num_frm= 5,
                                   max_num_obj=16,
                                   num_classes=67,
                                   rpn_feature_dims=4096,
                                   num_threads=6,
                                   dataset_name="cvpr")

        val_set = YC2_test_data(label_file=args.yc2_val_file,
                                gt_box_file=args.yc2_val_gt_box,
                                class_file=args.yc2_class_file,
                                data_split='validation',
                                rpn_proposal_root=args.cvpr_proposal_root,
                                roi_pooled_feat_root=args.cvpr_rpn_feat_root,
                                num_proposals=20,
                                max_num_obj=16,
                                num_classes=67,
                                rpn_feature_dims=4096,
                                dataset_name="cvpr")
        test_set = YC2_test_data(label_file=args.yc2_test_file,
                                 gt_box_file=args.yc2_test_gt_box,
                                 class_file=args.yc2_class_file,
                                 data_split='testing',
                                 rpn_proposal_root=args.cvpr_proposal_root,
                                 roi_pooled_feat_root=args.cvpr_rpn_feat_root,
                                 num_proposals=20,
                                 max_num_obj=16,
                                 num_classes=67,
                                 rpn_feature_dims=4096,
                                 dataset_name="cvpr")
        return train_set, val_set, test_set
    else:
        exit()


def get_cmd():
    # paras:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="CVPR", help="CVPR (4096-dim) YC2 dataset")
    parser.add_argument("--model_type", default="full", help="model types")
    parser.add_argument("--train_test", default="test", help="training or testing")
    parser.add_argument("--model_dir", default="./checkpoint/ModelBest.pkl", help="model directory")
    parser.add_argument("--initialize", default=None, help="initialization")
    parser.add_argument("--train_mode", default="ACL_Spatial", help="train mode")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    paras = get_cmd()
    args = TfConfig()

    args.feature = paras.dataset_name
    main(args, paras)
