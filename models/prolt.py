import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MLP
import numpy as np
from .embeddings import _Py, _Vs_Vo, _Ps_Po

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class CZSL(nn.Module):
    def __init__(self, dset, args):
        super(CZSL, self).__init__()
        self.args = args
        self.dset = dset
        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs
        def remove_from_B(A, B):
            A_set = set(A.tolist())
            B_set = set(B.tolist())
            new_B_set = B_set - A_set
            new_B = torch.tensor(list(new_B_set), dtype=torch.long)
            return new_B
        def union_of_AB(A, B):
            combined = torch.cat([A, B], dim=0)
            union = torch.unique(combined)
            return union
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.scale = args.tem
        self.train_forward = self.train_forward_closed
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.pairs = dset.pairs

        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)
        if args.nlayers:
            self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                      dropout=self.args.dropout,
                                      norm=self.args.norm, layers=layers)
        if self.args.train_only:
            train_idx = []
            test_idx = []
            val_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current])
            self.train_idx = torch.LongTensor(train_idx).to(device)

            for current in dset.val_pairs:
                val_idx.append(dset.all_pair2idx[current])
            self.val_idx = torch.LongTensor(val_idx).to(device)
            self.val_idx = remove_from_B(self.train_idx,self.val_idx)

            for current in dset.test_pairs:
                test_idx.append(dset.all_pair2idx[current])
            self.test_idx = torch.LongTensor(test_idx).to(device)
            self.test_idx = remove_from_B(self.train_idx,self.test_idx)
            self.unseen_idx = union_of_AB(self.val_idx, self.test_idx)

        self.label_smooth = torch.zeros(self.num_pairs, self.num_pairs)

        for i in range(self.num_pairs):
            for j in range(self.num_pairs):
                if self.pairs[j][1] == self.pairs[i][1]:
                    self.label_smooth[i, j] = self.label_smooth[i, j] + 2
                if self.pairs[j][0] == self.pairs[i][0]:
                    self.label_smooth[i, j] = self.label_smooth[i, j] + 1
        self.label_smooth = self.label_smooth[:, self.train_idx]
        self.label_smooth = self.label_smooth[self.train_idx, :]

        K_1 = (self.label_smooth == 1).sum(dim=1)
        K_2 = (self.label_smooth == 2).sum(dim=1)
        K = K_1 + 2 * K_2
        self.epi = self.args.smooth
        template = torch.ones_like(self.label_smooth)/K
        template = template * self.epi
        self.label_smooth = self.label_smooth*template
        for i in range(self.label_smooth.shape[0]):
            self.label_smooth[i,i] = 1 - (self.epi)
        self.label_smooth = self.label_smooth.to(device)

        self.Py = _Py(self.dset,self.args)
        self.VsVo = _Vs_Vo(self.dset,self.args)
        self.PsPo = _Ps_Po(self.dset,self.args)

        self.obj2pairs = self.create_obj_pairs().to(device)
        self.attr2pairs = self.create_attr_pairs().to(device)

        if args.train_only:
            self.obj2pairs = self.obj2pairs[:,self.train_idx].to(device)
            self.attr2pairs = self.attr2pairs[:,self.train_idx].to(device)

        self.p_log = dict()
    def cross_entropy(self, logits, label):
        logits = F.log_softmax(logits, dim=-1)
        loss = -(logits * label).sum(-1).mean()
        return loss

    def create_obj_pairs(self):
        obj_matrix = torch.zeros(self.num_objs,self.num_pairs)
        for i in range(self.num_objs):
            for j in range(self.num_pairs):
                if self.dset.objs[i] == self.pairs[j][1]:
                    obj_matrix[i,j] = 1
        return obj_matrix
    def create_attr_pairs(self):
        obj_matrix = torch.zeros(self.num_attrs,self.num_pairs)
        for i in range(self.num_attrs):
            for j in range(self.num_pairs):
                if self.dset.attrs[i] == self.pairs[j][0]:
                    obj_matrix[i,j] = 1
        return obj_matrix
    def create_S2U(self):
        S2U_matrixs = torch.zeros(len(self.train_pairs), self.num_pairs)
        S2U_matrixu = torch.zeros(len(self.train_pairs), self.num_pairs)
        S2U_matrixp = torch.zeros(len(self.train_pairs), self.num_pairs)
        for i in range(len(self.train_pairs)):
            for j in range(self.num_pairs):
                if self.dset.train_pairs[i][0] == self.pairs[j][0] and self.dset.train_pairs[i][1] != self.pairs[j][1] \
                        and j not in self.train_idx:
                    S2U_matrixs[i,j] = 1
                if self.dset.train_pairs[i][1] == self.pairs[j][1] and self.dset.train_pairs[i][0] != self.pairs[j][0] \
                        and j not in self.train_idx:
                    S2U_matrixu[i,j] = 1
                if self.dset.train_pairs[i][1] == self.pairs[j][1] and self.dset.train_pairs[i][0] == self.pairs[j][0]:
                    S2U_matrixp[i,j] = 1
        return S2U_matrixs, S2U_matrixu, S2U_matrixp

    def Classifiers(self, type, feat, attr, objs):
        if type == 'y':
            pyy = self.Py(attr, objs).permute(1, 0)
            logits = torch.mm(feat, pyy)
        elif type == 'so':
            logits = {}
            xs, xo = self.VsVo(feat)
            pss, poo = self.PsPo(attr,objs)
            logits['s'] = torch.mm(xs, pss.permute(1, 0))
            logits['o'] = torch.mm(xo, poo.permute(1, 0))
        else:
            raise ValueError("Please enter the correct classifier type")
        return logits
    def get_x(self, img):
        if self.args.nlayers:
            xy = self.image_embedder(img)
        else:
            xy = img
        xso = img.clone()

        if self.args.P_y == 'MLP':
            xy = F.normalize(xy, dim=1)
        xso = F.normalize(xso, dim=-1)
        return xy, xso
    def freeze_model(self):
        for p in self.Py.parameters():
            p.requires_grad = False
        if self.args.nlayers:
            for p in self.image_embedder.parameters():
                p.requires_grad = False
    def unfreeze_model(self):
        for p in self.Py.parameters():
            p.requires_grad = True
        if self.args.nlayers:
            for p in self.image_embedder.parameters():
                p.requires_grad = True

    def val_forward(self, x, prior):
        img = x[0]
        img = F.avg_pool2d(img, kernel_size=7).view(-1, self.dset.feat_dim)
        feats_y, feats_so = self.get_x(img)

        pred_so = self.Classifiers('so', feats_so, self.val_attrs,self.val_objs)
        pred_so['s'] = F.softmax(pred_so['s'], dim=-1)
        pred_so['o'] = F.softmax(pred_so['o'], dim=-1)

        if prior == False:
            pred_y = self.Classifiers('y', feats_y, self.val_attrs, self.val_objs)
            s = pred_y + self.args.eta * torch.log((pred_so['s'] * pred_so['o']))
        else:
            s = self.args.eta * torch.log((pred_so['s'] * pred_so['o']))

        score = F.softmax(s, dim=-1)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return None, scores

    def train_forward_closed(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        img = F.avg_pool2d(img, kernel_size=7).view(-1, self.dset.feat_dim)
        smoothed_labels = self.label_smooth[pairs]

        feats_y, feats_so = self.get_x(img)

        k_so = self.args.eta * torch.log(self.p_log['attr'] * self.p_log['objs'])
        pred_y = self.Classifiers('y', feats_y, self.train_attrs, self.train_objs)
        L_cls = self.cross_entropy(self.scale * (pred_y) + self.args.alpha * k_so.detach(), smoothed_labels)

        pred_so = self.Classifiers('so', feats_so, self.uniq_attrs,self.uniq_objs)
        L_s = F.cross_entropy(self.scale * pred_so['s'], attrs)
        L_o = F.cross_entropy(self.scale * pred_so['o'], objs)

        loss = L_cls + L_s + L_o

        with torch.no_grad():
            E = []
            E.append(pred_so['s'].detach()@self.attr2pairs)
            E.append(pred_so['o'].detach()@self.obj2pairs)

            mask_a = torch.zeros_like(E[0])
            mask_o = torch.zeros_like(E[0])
            mask_a[:,pairs] = 1
            mask_o[:,pairs] = 1
            E[0] = E[0] * mask_a
            E[1] = E[1] * mask_o
        return loss, E

    def train_forward_prior(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        img = F.avg_pool2d(img, kernel_size=7).view(-1, self.dset.feat_dim)
        feats_y, feats_so = self.get_x(img)

        pred_so = self.Classifiers('so', feats_so, self.uniq_attrs,self.uniq_objs)
        L_s = F.cross_entropy(self.scale * pred_so['s'], attrs)
        L_o = F.cross_entropy(self.scale * pred_so['o'], objs)

        loss = L_s + L_o

        with torch.no_grad():
            E = []
            E.append(pred_so['s'].detach()@self.attr2pairs)
            E.append(pred_so['o'].detach()@self.obj2pairs)

            mask_a = torch.zeros_like(E[0])
            mask_o = torch.zeros_like(E[0])
            mask_a[:, pairs] = 1
            mask_o[:, pairs] = 1
            E[0] = E[0] * mask_a
            E[1] = E[1] * mask_o
        return loss, E

    def forward(self, x, prior = False):
        if self.training:
            if prior == False:
                loss, pred = self.train_forward_closed(x)
            else:
                loss, pred = self.train_forward_prior(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x, prior)
        return loss, pred



