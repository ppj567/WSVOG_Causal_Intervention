import torch
import torch.nn as nn
import torch.nn.functional as F

from Attention_mm import MultiHeadAttention
import numpy as np
import os
from memory import MemoryBank
import random
import json

random.seed(123)
def pause():
    programPause = input("Press the <ENTER> key to continue...")

class WSTOG(nn.Module):
    def __init__(self, mem_slots_num=10, input_feature_dims=4096, input_label_dims=67, embedding_size=512, drop_crob=0.9, batch_size=20, max_num_obj=16, num_classes=67, dataset_feat="CVPR", beta=0.5):

        'build input parameters'
        super(WSTOG, self).__init__()

        self.mem_slots_num = mem_slots_num
        self.embedding_size = embedding_size
        self.train_drop_crob = drop_crob
        self.input_label_dims = input_label_dims
        self.input_feature_dims = input_feature_dims
        self.batch_size =batch_size
        self.dataset_feat = dataset_feat
        self.beta = beta

        self.num_class = num_classes + 1
        self.GRAPH_LAYERS = 1

        self.mem_size = 100

        # spatial memory banks
        rand_idx = torch.tensor(random.sample([i for i in range(700)],self.mem_size))
        self.memorybank = MemoryBank(memory_num=self.mem_size, memory_dim=self.input_feature_dims, momentum=0.9)
        self.memorybank.set_weight(torch.load('mem700.pkl')[rand_idx].cuda(),index=torch.tensor([i for i in range(self.mem_size)]).cuda())

        rand_idx = torch.tensor(random.sample([i for i in range(700)],self.mem_size))
        self.memorybank2 = MemoryBank(memory_num=self.mem_size, memory_dim=self.input_feature_dims, momentum=0.9)
        self.memorybank2.set_weight(torch.load('mem700.pkl')[rand_idx].cuda( ),index=torch.tensor([i for i in range(self.mem_size)]).cuda())

        # temporal memory banks
        self.temporal_mem_size = 1900
        rand_idx = torch.tensor(random.sample([i for i in range(1919)],self.temporal_mem_size))
        self.temporal_memorybank = MemoryBank(memory_num=self.temporal_mem_size, memory_dim=self.input_feature_dims*20, momentum=0.9)
        self.temporal_memorybank.set_weight(torch.load('frame_mem1919x20x4096.pkl')[rand_idx].cuda(),index=torch.tensor([i for i in range(self.temporal_mem_size)]).cuda())

        rand_idx = torch.tensor(random.sample([i for i in range(1919)],self.temporal_mem_size))
        self.temporal_memorybank2 = MemoryBank(memory_num=self.temporal_mem_size, memory_dim=self.input_feature_dims*20, momentum=0.9)
        self.temporal_memorybank2.set_weight(torch.load('frame_mem1919x20x4096.pkl')[rand_idx].cuda(),index=torch.tensor([i for i in range(self.temporal_mem_size)]).cuda())


        We = torch.eye(num_classes + 1)
        We[0,:]= torch.zeros(num_classes + 1).cuda().float()

        self.one_hot_ = nn.Embedding.from_pretrained(We, freeze=True)
        self.entity_embeddings = nn.Linear(self.num_class, self.embedding_size)
        nn.init.xavier_uniform_(self.entity_embeddings.weight, gain=nn.init.calculate_gain('relu'))

        self.temp_attention = nn.Sequential(
            nn.Linear(4*self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(self.embedding_size, 1))

        self.dropout_atten = nn.Dropout(p=0.1)

        self.dropout_query = nn.Dropout(p = 0.1)
        self.dropout_ref = nn.Dropout(p = 0.1)
        self.dropout_att = nn.Dropout(p = 0.1)

        self.HIDDEN_SIZE = 512
        self.FF_SIZE = 512
        self.DROPOUT_R = 0.1
        self.MULTI_HEAD = 8
        self.HIDDEN_SIZE_HEAD = 64

        self.multihead_attn = MultiHeadAttention(self.input_feature_dims, self.embedding_size, num_heads=8, dropout=0.1)
        self.randmem = []
        self.K = 10
        print('spatialK= ',self.K)
        self.temporal_K = 1
        print('tempK= ',self.temporal_K)

        self.balance_beta = 0.9999
        with open("class_times.json") as f:
            self.num_per_class = json.load(f)

        with open("class_times.json") as f:
            self.num_per_class0 = json.load(f)
        self.num_per_class0 = torch.Tensor(list(self.num_per_class0.values()))
        self.num_per_class0 = self.num_per_class0/torch.sum(self.num_per_class0)
        self.num_per_class0 = self.num_per_class0.cuda()
        self.num_per_class.pop('63')
        self.num_per_class.pop('64')
        self.num_per_class.pop('65')
        self.num_per_class.pop('66')
        self.num_per_class = torch.Tensor(list(self.num_per_class.values()))
        self.effective_num = 1.0 - torch.pow(self.balance_beta, self.num_per_class)
        self.balance_weights = (1.0 - self.balance_beta) / self.effective_num
        self.balance_weights = self.balance_weights / torch.sum(self.balance_weights) * (num_classes - 4)
        a=list(self.balance_weights)
        new_p = 0.2
        a.append(torch.tensor(new_p))
        a.append(torch.tensor(new_p))
        a.append(torch.tensor(new_p))
        a.append(torch.tensor(new_p))
        b = torch.Tensor(a)
        self.balance_weights = b/torch.sum(b) * num_classes
        self.balance_weights = torch.cat([torch.zeros(1,1), self.balance_weights.unsqueeze(-1)],dim=0).squeeze(-1)
        self.balance_weights = self.balance_weights * 21249/11024.1855

        self.objectness_cls = nn.Sequential(
            nn.Linear(self.input_feature_dims, 512),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        if self.dataset_feat=="ANET":
            self.W_glove = torch.load('Anet_Glove_dict.pkl').cuda()
        elif self.dataset_feat=="CVPR":
            self.W_glove = torch.load('YC2_glove_dict.pth').cuda()
        else:
            self.W_glove = torch.load('YC2_glove_dict.pth').cuda() # we do not use it in RoboWatch
        self.glove_embedding = nn.Embedding.from_pretrained(self.W_glove, freeze=True)
        self.glove_linear = nn.Linear(300, self.embedding_size)
        nn.init.xavier_uniform_(self.glove_linear.weight, gain=nn.init.calculate_gain('relu'))

        self.glove_linear512 = nn.Linear(300, self.embedding_size)
        nn.init.xavier_uniform_(self.glove_linear512.weight, gain=nn.init.calculate_gain('relu'))



    def _select_objectness(self, QR_p, embedding_p, input_pos_label, input_pos_feature):
        QR_p = QR_p.detach()
        K = 1
        q = 0.7

        _, high_region_indices = torch.topk(QR_p,k=K,dim=-1)
        _, low_region_indices = torch.topk(QR_p,k=20-K,dim=-1,largest=False)

        rand_idx = np.random.randint(0,20-K,size=high_region_indices.size())
        low_region_indices = torch.gather(low_region_indices, -1, torch.tensor(rand_idx).cuda())


        if self.dataset_feat=="CVPR":
            R = F.normalize(input_pos_feature, p=2, dim=-1)

        else:
            R = input_pos_feature

        object_select = torch.ne(input_pos_label, 0)
        object_num = torch.sum(object_select, dim=1)

        R = R.detach()


        obj_embs = embedding_p['obj_labels']
        q_shape = obj_embs.size()
        r_shape = input_pos_feature.size()

        Q = obj_embs
        R_shape = R.size()
        R = R.view(-1,R.shape[-1])

        R = R.view(R_shape)

        i = torch.arange(QR_p.shape[0]).cuda()
        j = torch.arange(QR_p.shape[2]).cuda()
        k = torch.arange(QR_p.shape[1]).cuda()
        l = torch.arange(K).cuda()
        grids = list(torch.meshgrid(i, k, j, l))

        pre_indices = torch.stack([
            grids[0],
            grids[2]
        ], dim=-1)
        indices = torch.cat([pre_indices, high_region_indices.unsqueeze(-1)], -1)

        splits = [split.squeeze(-1).long() for split in indices.split(1, dim=-1)]
        R_high=(R[splits[0], splits[1], splits[2]]).contiguous()

        indices = torch.cat([pre_indices, low_region_indices.unsqueeze(-1)], -1)
        splits = [split.squeeze(-1).long() for split in indices.split(1, dim=-1)]
        R_low=(R[splits[0], splits[1], splits[2]]).contiguous()

        R_high = (object_select.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * R_high).squeeze(-2)
        R_low = (object_select.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * R_low).squeeze(-2)

        R_high_result = self.objectness_cls(R_high)
        R_high_result = (1 - (R_high_result + 1e-7)**q)/q
        R_low_result = self.objectness_cls(R_low)
        R_low_result = (1 - (1 - R_low_result + 1e-7)**q)/q
        Generalized_CE_high = torch.mean((object_select.unsqueeze(-1).unsqueeze(-1) * R_high_result).squeeze(-1),-1)
        Generalized_CE_high = torch.mean(torch.div(torch.sum(Generalized_CE_high, dim = 1), object_num))
        Generalized_CE_low = torch.mean((object_select.unsqueeze(-1).unsqueeze(-1) * R_low_result).squeeze(-1),-1)
        Generalized_CE_low = torch.mean(torch.div(torch.sum(Generalized_CE_low, dim = 1), object_num))
        Generalized_CE_loss = Generalized_CE_high + Generalized_CE_low



        z=0
        return Generalized_CE_loss






    def _temporal_acl(self, QR_frm_weight_p, embedding_p, input_pos_label, input_pos_feature):
        QR_frm_weight_p = QR_frm_weight_p.detach()
        frame_num = QR_frm_weight_p.shape[-1]
        K = self.temporal_K

        _, high_region_indices = torch.topk(QR_frm_weight_p,k=K,dim=-1)
        _, low_region_indices = torch.topk(QR_frm_weight_p,k=frame_num-K,dim=-1,largest=False)

        rand_idx = np.random.randint(0,frame_num-K,size=high_region_indices.size())
        low_region_indices = torch.gather(low_region_indices, -1, torch.tensor(rand_idx).cuda())

        if self.dataset_feat=="CVPR":
            R = F.normalize(input_pos_feature, p=2, dim=-1)

        else:
            R = input_pos_feature

        object_select = torch.ne(input_pos_label, 0)

        obj_embs = embedding_p['obj_labels']
        q_shape = obj_embs.size()
        r_shape = input_pos_feature.size()

        Q = obj_embs
        R_shape = R.size()

        R_maxpool = F.normalize(torch.max(R,dim=2)[0].view(-1,R_shape[-1]),dim=-1)

        temporal_memorybank_maxpool = F.normalize(torch.max(self.temporal_memorybank.memory.view(self.temporal_mem_size,-1,R_shape[-1]),dim=1)[0],dim=-1)
        temporal_memorybank2_maxpool = F.normalize(torch.max(self.temporal_memorybank2.memory.view(self.temporal_mem_size,-1,R_shape[-1]),dim=1)[0],dim=-1)
        sim_idx = (torch.max(torch.mm(R_maxpool, temporal_memorybank_maxpool.permute(1,0)),dim=-1)[1])
        Rp_updated = self.temporal_memorybank.memory[sim_idx].view(R_shape) # low -> positive
        sim_idx = (torch.max(torch.mm(R_maxpool, temporal_memorybank2_maxpool.permute(1,0)),dim=-1)[1])
        Rn_updated = self.temporal_memorybank2.memory[sim_idx].view(R_shape) # high -> negative

        R = R.view(R_shape)
        Rn = R.unsqueeze(0).clone().repeat(q_shape[1],1,1,1,1)
        Rp = R.unsqueeze(0).clone().repeat(q_shape[1],1,1,1,1)

        i = torch.arange(QR_frm_weight_p.shape[0]).cuda()
        k = torch.arange(QR_frm_weight_p.shape[1]).cuda()
        l = torch.arange(self.temporal_K).cuda()
        grids = list(torch.meshgrid(i, k, l))

        pre_indices = torch.stack([
            grids[0]
        ], dim=-1)
        indices = torch.cat([pre_indices, high_region_indices.unsqueeze(-1)], -1)

        splits = [split.squeeze(-1).long() for split in indices.split(1, dim=-1)]
        R_high=(R[splits[0], splits[1]]).permute(1,0,2,3,4).contiguous()

        indices = torch.cat([pre_indices, low_region_indices.unsqueeze(-1)], -1)
        splits = [split.squeeze(-1).long() for split in indices.split(1, dim=-1)]
        R_low=(R[splits[0], splits[1]]).permute(1,0,2,3,4).contiguous()


        new_grids = list(torch.meshgrid(
            i,
            k,
            l
        ))

        new_pre_indices = torch.stack([
            new_grids[0],
            new_grids[1]
        ], dim=-1)
        new_indices = torch.cat([new_pre_indices, high_region_indices.unsqueeze(-1)], -1)
        new_splits = [split.squeeze(-1).long() for split in new_indices.split(1, dim=-1)]
        Rn = Rn.permute(1,0,2,3,4)
        Rn[new_splits[0], new_splits[1], new_splits[2]] = Rn_updated[splits[0], splits[1]]
        Rn = Rn.permute(1,0,2,3,4)

        new_indices = torch.cat([new_pre_indices, low_region_indices.unsqueeze(-1)], -1)
        new_splits = [split.squeeze(-1).long() for split in new_indices.split(1, dim=-1)]
        Rp = Rp.permute(1,0,2,3,4)
        Rp[new_splits[0], new_splits[1], new_splits[2]] = Rp_updated[splits[0], splits[1]]
        Rp = Rp.permute(1,0,2,3,4)

        R_high = R_high.view(-1,r_shape[2]*R_shape[3])
        R_low = R_low.view(-1,r_shape[2]*R_shape[3])
        R_high_maxpooling = F.normalize(torch.max(R_high.view(-1,r_shape[2],R_shape[3]),dim=1)[0],dim=-1)
        R_low_maxpooling = F.normalize(torch.max(R_low.view(-1,r_shape[2],R_shape[3]),dim=1)[0],dim=-1)
        high_sim = (torch.max(torch.mm(R_high_maxpooling, temporal_memorybank2_maxpool.permute(1,0)),dim=-1)[1])
        low_sim = (torch.max(torch.mm(R_low_maxpooling, temporal_memorybank_maxpool.permute(1,0)),dim=-1)[1])


        # update memory bank
        for n in range(self.temporal_memorybank.memory.shape[0]):
            cnt = 0
            region_mask_high=torch.logical_and((high_sim.unsqueeze(-1) == n).view(q_shape[1],r_shape[0],K),(object_select.permute(1,0).unsqueeze(-1)))
            num_all_high = torch.sum(region_mask_high.view(-1))
            if num_all_high > 0:
                high_mean = ((high_sim.unsqueeze(-1) == n) * R_high).view(q_shape[1],r_shape[0],K,r_shape[2]*R_shape[3]) * (object_select.permute(1,0).unsqueeze(-1).unsqueeze(-1))
                high_mean = torch.sum(high_mean.view(-1,r_shape[2]*R_shape[3]),dim=0,keepdim=True) / num_all_high
                self.temporal_memorybank2.update_weight(high_mean, n)
                cnt += 1
            else:
                high_mean = 0

            region_mask_low=torch.logical_and((low_sim.unsqueeze(-1) == n).view(q_shape[1],r_shape[0],K),(object_select.permute(1,0).unsqueeze(-1)))
            num_all_low = torch.sum(region_mask_low.view(-1))
            if num_all_low > 0:
                low_mean = ((low_sim.unsqueeze(-1) == n) * R_low).view(q_shape[1],r_shape[0],K,r_shape[2]*R_shape[3]) * (object_select.permute(1,0).unsqueeze(-1).unsqueeze(-1))
                low_mean = torch.sum(low_mean.view(-1,r_shape[2]*R_shape[3]),dim=0,keepdim=True) / num_all_low
                self.temporal_memorybank.update_weight(low_mean, n)
                cnt += 1
            else:
                low_mean = 0


        return Q, Rn, Rp







    def _spatial_acl(self, QR_p, embedding_p, input_pos_label, input_pos_feature):
        QR_p = QR_p.detach()
        K = self.K

        if self.dataset_feat=="CVPR":
            R = F.normalize(input_pos_feature, p=2, dim=-1)
        else:
            R = input_pos_feature

        _, high_region_indices = torch.topk(QR_p,k=K,dim=-1)
        _, low_region_indices = torch.topk(QR_p,k=20-K,dim=-1,largest=False)


        rand_idx = np.random.randint(0,20-K,size=high_region_indices.size())
        low_region_indices = torch.gather(low_region_indices, -1, torch.tensor(rand_idx).cuda())

        object_select = torch.ne(input_pos_label, 0)


        obj_embs = embedding_p['obj_labels']
        q_shape = obj_embs.size()
        r_shape = input_pos_feature.size()

        Q = obj_embs
        R_shape = R.size()
        R = R.view(-1,R.shape[-1])
        sim_idx = (torch.max(torch.mm(R, self.memorybank.memory.permute(1,0)),dim=-1)[1])
        Rp_updated = self.memorybank.memory[sim_idx].view(R_shape) #low -> positive
        sim_idx = (torch.max(torch.mm(R, self.memorybank2.memory.permute(1,0)),dim=-1)[1])
        Rn_updated = self.memorybank2.memory[sim_idx].view(R_shape) #high -> negative

        R = R.view(R_shape)
        Rn = R.unsqueeze(0).clone().repeat(q_shape[1],1,1,1,1)
        Rp = R.unsqueeze(0).clone().repeat(q_shape[1],1,1,1,1)

        i = torch.arange(QR_p.shape[0]).cuda()
        j = torch.arange(QR_p.shape[2]).cuda()
        k = torch.arange(QR_p.shape[1]).cuda()
        l = torch.arange(self.K).cuda()
        grids = list(torch.meshgrid(i, k, j, l))

        pre_indices = torch.stack([
            grids[0],
            grids[2]
        ], dim=-1)
        indices = torch.cat([pre_indices, high_region_indices.unsqueeze(-1)], -1)

        splits = [split.squeeze(-1).long() for split in indices.split(1, dim=-1)]
        R_high=(R[splits[0], splits[1], splits[2]]).permute(1,0,2,3,4).contiguous()

        indices = torch.cat([pre_indices, low_region_indices.unsqueeze(-1)], -1)
        splits = [split.squeeze(-1).long() for split in indices.split(1, dim=-1)]
        R_low=(R[splits[0], splits[1], splits[2]]).permute(1,0,2,3,4).contiguous()


        new_grids = list(torch.meshgrid(
            i,
            k,
            j,
            l
        ))

        new_pre_indices = torch.stack([
            new_grids[0],
            new_grids[1],
            new_grids[2]
        ], dim=-1)
        new_indices = torch.cat([new_pre_indices, high_region_indices.unsqueeze(-1)], -1)
        new_splits = [split.squeeze(-1).long() for split in new_indices.split(1, dim=-1)]
        Rn = Rn.permute(1,0,2,3,4)
        Rn[new_splits[0], new_splits[1], new_splits[2], new_splits[3]] = Rn_updated[splits[0], splits[1], splits[2]]
        Rn = Rn.permute(1,0,2,3,4)

        new_indices = torch.cat([new_pre_indices, low_region_indices.unsqueeze(-1)], -1)
        new_splits = [split.squeeze(-1).long() for split in new_indices.split(1, dim=-1)]
        Rp = Rp.permute(1,0,2,3,4)
        Rp[new_splits[0], new_splits[1], new_splits[2], new_splits[3]] = Rp_updated[splits[0], splits[1], splits[2]]
        Rp = Rp.permute(1,0,2,3,4)


        R_high = R_high.view(-1,R_shape[3])
        R_low = R_low.view(-1,R_shape[3])
        high_sim = (torch.max(torch.mm(R_high, self.memorybank2.memory.permute(1,0)),dim=-1)[1])
        low_sim = (torch.max(torch.mm(R_low, self.memorybank.memory.permute(1,0)),dim=-1)[1])


        # update memory bank
        for n in range(self.memorybank.memory.shape[0]):
            cnt = 0
            region_mask_high=torch.logical_and((high_sim.unsqueeze(-1) == n).view(q_shape[1],r_shape[0],r_shape[1],K),(object_select.permute(1,0).unsqueeze(-1).unsqueeze(-1)))
            num_all_high = torch.sum(region_mask_high.view(-1))
            if num_all_high > 0:
                high_mean = ((high_sim.unsqueeze(-1) == n) * R_high).view(q_shape[1],r_shape[0],r_shape[1],K,R_shape[3]) * (object_select.permute(1,0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                high_mean = torch.sum(high_mean.view(-1,R_shape[3]),dim=0,keepdim=True) / num_all_high
                self.memorybank2.update_weight(high_mean, n)
                cnt += 1
            else:
                high_mean = 0

            region_mask_low=torch.logical_and((low_sim.unsqueeze(-1) == n).view(q_shape[1],r_shape[0],r_shape[1],K),(object_select.permute(1,0).unsqueeze(-1).unsqueeze(-1)))
            num_all_low = torch.sum(region_mask_low.view(-1))
            if num_all_low > 0:
                low_mean = ((low_sim.unsqueeze(-1) == n) * R_low).view(q_shape[1],r_shape[0],r_shape[1],K,R_shape[3]) * (object_select.permute(1,0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                low_mean = torch.sum(low_mean.view(-1,R_shape[3]),dim=0,keepdim=True) / num_all_low
                self.memorybank.update_weight(low_mean, n)
                cnt += 1
            else:
                low_mean = 0

        return Q, Rn, Rp




    def _feat_extract(self, Q, R,  is_training):
        embedding = {}
        object_select = torch.ne(Q, 0)
        obj = self.one_hot_(Q)
        obj_embs = self.entity_embeddings(obj)

        obj_embs = self.glove_embedding(Q)
        obj_embs = self.glove_linear(obj_embs)

        embedding['obj_idx'] = Q

        if self.dataset_feat=="CVPR":
            R = F.normalize(R, p=2, dim=-1)

        obj_embs = self.dropout_query(torch.relu(obj_embs))
        obj_embs = F.normalize(obj_embs, p=2, dim=-1)

        embedding['obj_labels'] = obj_embs

        obj_sent_embed = torch.zeros(Q.size(0), obj_embs.size(-1))

        for x in range(Q.size(0)):
            obj_sent_embed[x] = obj_embs[x, object_select[x]].max(0)[0]

        embedding['obj_sent_embed'] = obj_sent_embed.type(torch.cuda.FloatTensor)

        rgb_feats, attend_out = self.multihead_attn(R, self.dataset_feat, is_training)
        rgb_feats = self.dropout_ref(torch.tanh(rgb_feats))
        embedding['rgb_feats_att'] = torch.tanh(attend_out)
        embedding['rgb_feats'] = rgb_feats

        rgb_feats = F.normalize(rgb_feats, p=2, dim=-1)

        frm_emb = torch.max(rgb_feats, dim=-2)[0]
        embedding['rgb_feats_frm']  = frm_emb

        return embedding


    def _compute_score(self, Embedding_Q, Embedding_R, R, is_training, causal=False, test_object_num=None, original_frm_num=None):

        q_shape = Embedding_Q["obj_labels"].size()
        r_shape = R.size()

        Q = Embedding_Q["obj_labels"]
        R0 = self.dropout_att(torch.add(Embedding_R['rgb_feats'], Embedding_R['rgb_feats_att']))
        R0shape = R0.shape
        if causal:
            Gobj_embs = self.glove_linear512(self.glove_embedding.weight[1:])
            Gobj_embs = self.dropout_query(Gobj_embs)

            #Attention-based fusion strategy
            # Gobj_embs = F.normalize(Gobj_embs,p=2,dim=-1)
            # R0 = F.normalize(R0, p=2, dim=-1)
            # # attention
            # R1 = torch.mean(R0,-2)[0]
            # R1shape = R1.shape
            # R1 = R1.view(-1,R1shape[-1])
            # # R0 = R0.view(-1,R0shape[-1])
            # R1 = F.normalize(R1,p=2,dim=-1)
            # attention = torch.mm(self.Wy(R1), self.Wz(Gobj_embs).t()) / (self.embedding_size ** 0.5)
            # attention = F.softmax(attention, -1)
            # z_hat = torch.mm(attention, Gobj_embs*self.num_per_class0.unsqueeze(-1))
            # # z_hat = self.att_bn(z_hat)
            # z_hat = z_hat.view(R1shape)
            # R0 = R0.view(R0shape)
            # R0 = R0 + z_hat.unsqueeze(-2)
            R0 = 0.7*R0+ torch.sum((Gobj_embs.unsqueeze(1).unsqueeze(1).unsqueeze(1)) * (self.num_per_class0).view(-1,1,1,1,1).cuda(), dim=0)/4*0.3


        R0 = F.normalize(R0, p=2, dim=-1)
        R1 =R0.detach()

        Q = Q.view(q_shape[0], q_shape[1], 1, 1, q_shape[2])
        R0 = R0.view(r_shape[0], 1, r_shape[1], r_shape[2], -1)

        QR = torch.mul(Q, R0).sum(dim=-1)

        if is_training:
            object_select = torch.ne(Embedding_Q["obj_idx"], 0).type(torch.cuda.FloatTensor)
            QR_frm_weight= self.attention(Embedding_Q, Embedding_R)
            QR_frm =torch.max(QR, dim=3)[0]
            all_score = torch.sum(torch.mul(QR_frm, QR_frm_weight), dim=-1)
            object_num = torch.sum(object_select, dim=1)
            select_score = torch.mul(all_score, object_select)
            score = torch.div(torch.sum(select_score, dim = 1), object_num)
            consis_loss = 0
            return score, select_score, QR, consis_loss, QR_frm_weight
        else:
            QR = torch.sigmoid(QR)
            QR = QR.transpose(0,1)
            QR = QR.contiguous().view(1, q_shape[1], q_shape[0]*r_shape[1], r_shape[2])
            QR = QR[:,:,:original_frm_num]

            score = QR[:, :test_object_num, :, :]

        return score


    def _compute_acl_score(self, QR_frm_weight_p, QR, embedding_q, Q, embedding_counter_n, embedding_counter_p, input_pos_label, input_pos_feature):
        q_shape = Q.size()
        r_shape = input_pos_feature.size()
        embedding_rp = {}
        embedding_rn = {}

        rgb_feats_p, attend_out_p = self.multihead_attn(embedding_counter_p, self.dataset_feat, True)
        rgb_feats_p = self.dropout_ref(torch.tanh(rgb_feats_p))
        attend_out_p = torch.tanh(attend_out_p)
        Rp = self.dropout_att(torch.add(rgb_feats_p, attend_out_p))
        Rp = F.normalize(Rp, p=2, dim=-1)

        frm_emb_p = torch.max(F.normalize(rgb_feats_p, p=2, dim=-1), dim=-2)[0]
        embedding_rp['rgb_feats_frm']  = frm_emb_p

        rgb_feats_n, attend_out_n = self.multihead_attn(embedding_counter_n, self.dataset_feat, True)
        rgb_feats_n = self.dropout_ref(torch.tanh(rgb_feats_n))
        attend_out_n = torch.tanh(attend_out_n)
        Rn = self.dropout_att(torch.add(rgb_feats_n, attend_out_n))
        Rn = F.normalize(Rn, p=2, dim=-1)

        R = self.dropout_att(torch.add(embedding_q['rgb_feats'], embedding_q['rgb_feats_att']))
        R = F.normalize(R, p=2, dim=-1).unsqueeze(1).expand(-1,q_shape[1],-1,-1,-1)

        frm_emb_n = torch.max(F.normalize(rgb_feats_n, p=2, dim=-1), dim=-2)[0]
        embedding_rn['rgb_feats_frm']  = frm_emb_n

        Q = Q.view(q_shape[0], q_shape[1], 1, 1, q_shape[2])
        Rp = Rp.permute(1,0,2,3,4)
        Rn = Rn.permute(1,0,2,3,4)


        QRp = torch.mul(Q, Rp).sum(dim=-1)
        QRn = torch.mul(Q, Rn).sum(dim=-1)

        object_select = torch.ne(input_pos_label, 0).type(torch.cuda.FloatTensor)
        object_num = torch.sum(object_select, dim=1)

        QRp_frm_weight= self.attention(embedding_q, embedding_rp)
        QRn_frm_weight= self.attention(embedding_q, embedding_rn)
        QRp_frm =torch.max(QRp, dim=3)[0]
        QRn_frm =torch.max(QRn, dim=3)[0]
        QR_frm = torch.max(QR,dim=3)[0]

        all_score_p = torch.sum(torch.mul(QRp_frm, QRp_frm_weight), dim=-1)
        all_score_n = torch.sum(torch.mul(QRn_frm, QRn_frm_weight), dim=-1)

        combined_weight = QR_frm_weight_p * QR_frm
        frame_idx = torch.max(combined_weight, dim=-1)[1]

        select_score_p = torch.mul(all_score_p, object_select)
        select_score_n = torch.mul(all_score_n, object_select)
        score_p = torch.div(torch.sum(select_score_p, dim = 1), object_num)
        score_n = torch.div(torch.sum(select_score_n, dim = 1), object_num)
        loss = (torch.mean(F.softplus(torch.sub(score_n, score_p).div(self.beta))))
        return loss




    def attention(self, Embedding_Q, Embedding_R):
        q_shape = Embedding_Q["obj_labels"].size()


        if Embedding_R['rgb_feats_frm'].ndim == 3:
            r_shape = Embedding_R['rgb_feats_frm'].size()
            R_frm_max = Embedding_R['rgb_feats_frm'].view(r_shape[0], 1, r_shape[1], r_shape[2]).expand([r_shape[0], q_shape[1], r_shape[1], self.embedding_size])
        else:
            R_frm_max = Embedding_R['rgb_feats_frm'].permute(1,0,2,3)
            r_shape = R_frm_max[:,0,:,:].size()
        Q_frm = Embedding_Q["obj_labels"].view(q_shape[0], q_shape[1],  1, q_shape[2]).expand([q_shape[0], q_shape[1], r_shape[1], q_shape[2]])
        Q_frm_max = Embedding_Q["obj_sent_embed"].view(q_shape[0], 1, 1, q_shape[2]).expand(q_shape[0], q_shape[1],  r_shape[1], q_shape[2])

        QR_frm_cat = torch.cat([torch.mul(R_frm_max, Q_frm), R_frm_max, Q_frm_max, Q_frm], dim=-1)

        QR_mv_attention = self.temp_attention(QR_frm_cat).squeeze(dim=-1)
        QR_frm_weight = F.softmax(QR_mv_attention, dim=-1)

        return QR_frm_weight


    def forward(self, input_pos_label, input_pos_feature, input_neg_label, input_neg_feature,  is_training, is_finetune):

        embedding_p = self._feat_extract(input_pos_label, input_pos_feature, is_training)
        embedding_n = self._feat_extract(input_neg_label, input_neg_feature, is_training)

        Sp1, select_score_p1, QR_p, consis_loss_p, QR_frm_weight_p = self._compute_score(embedding_p, embedding_p, input_pos_feature, is_training, True)
        Sp2, select_score_p2, QR_n, consis_loss_n, QR_frm_weight_n = self._compute_score(embedding_n, embedding_n, input_neg_feature, is_training, True)

        Srn, select_score_rn, QR_rn, consis_loss_rn, QR_frm_weight_rn = self._compute_score(embedding_p, embedding_n, input_neg_feature, is_training, True)
        Sqn, select_score_qn, QR_qn, consis_loss_qn, QR_frm_weight_qn = self._compute_score(embedding_n, embedding_p, input_pos_feature, is_training, True)

        total_loss = torch.add(self._log_loss(Sp1, Srn, Sqn), self._log_loss(Sp2, Srn, Sqn))

        #Spatial ACL
        Q, embedding_adv_n, embedding_adv_p = self._spatial_acl(QR_p, embedding_p, input_pos_label, input_pos_feature)
        loss_spatial = self._compute_acl_score(QR_frm_weight_p, QR_p, embedding_p, Q, embedding_adv_n, embedding_adv_p, input_pos_label, input_pos_feature)

        #Temporal ACL
        Q, embedding_adv_n, embedding_adv_p = self._temporal_acl(QR_frm_weight_p, embedding_p, input_pos_label, input_pos_feature)
        loss_temporal = self._compute_acl_score(QR_frm_weight_p, QR_p, embedding_p, Q, embedding_adv_n, embedding_adv_p, input_pos_label, input_pos_feature)

        return 0.7*total_loss + 0.3*(loss_temporal + loss_spatial)

    def _test_score(self, input_label, input_feature, test_object_num, original_frm_num,  is_training, causal):

        embedding_ = self._feat_extract(input_label, input_feature,  is_training)
        score = self._compute_score(embedding_, embedding_, input_feature, False, causal, test_object_num, original_frm_num)
        return score

    def _log_loss(self, Sp , Srn, Sqn):
        loss = torch.add(F.softplus(torch.sub(Srn, Sp).div(self.beta)), F.softplus(torch.sub(Sqn, Sp).div(self.beta)))/2.
        return torch.mean(loss)

    def _margin_loss(self, Sp , Srn, Sqn):
        loss = torch.add(torch.maximum(torch.zeros_like(Sp).cuda(), 0.6-Sp+Srn),torch.maximum(torch.zeros_like(Sp).cuda(), 0.6-Sp+Sqn))/2.
        return torch.mean(loss)

    def _log_lossv(self, Sp , Srn, Sqn):
        loss = torch.add(F.softplus(torch.sub(Srn, Sp).div(self.beta)), F.softplus(torch.sub(Sqn, Sp).div(self.beta)))/2.
        return torch.mean(loss)