import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import MLP
from .gcn import GCN, GCNII
from .word_embedding import load_word_embeddings
import scipy.sparse as sp

class _Py(nn.Module):
    def __init__(self,dset, args):
        super(_Py, self).__init__()
        self.dset = dset
        self.args = args
        device = 'cuda'.format(args.cuda_id) if torch.cuda.is_available() else 'cpu'

        if self.args.P_y == 'GCN':
            # Image Embedder
            self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
            self.pairs = dset.pairs
            if self.args.train_only:
                train_idx = []
                for current in dset.train_pairs:
                    train_idx.append(dset.all_pair2idx[current] + self.num_attrs + self.num_objs)
                self.train_idx = torch.LongTensor(train_idx).to(device)

            all_words = list(self.dset.attrs) + list(self.dset.objs)
            self.displacement = len(all_words)

            self.obj_to_idx = {word: idx for idx, word in enumerate(self.dset.objs)}
            self.attr_to_idx = {word: idx for idx, word in enumerate(self.dset.attrs)}

            if args.graph_init is not None:
                path = args.graph_init
                graph = torch.load(path)
                embeddings = graph['embeddings'].to(device)
                adj = graph['adj']
                self.embeddings = embeddings
            else:
                embeddings = self.init_embeddings(all_words).to(device)
                adj = self.adj_from_pairs()
                self.embeddings = embeddings

            hidden_layers = self.args.gr_emb

            self.gcn = GCN(adj, self.embeddings.shape[1], args.emb_dim, hidden_layers)

        elif self.args.P_y == 'MLP':
            '''get word vectors'''
            if args.emb_init == 'word2vec':
                word_vector_dim = 300
            elif args.emb_init == 'glove':
                word_vector_dim = 300
            elif args.emb_init == 'fasttext':
                word_vector_dim = 300
            else:
                word_vector_dim = 600

            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current])
            self.train_idx = torch.LongTensor(train_idx)

            self.attr_embedder = nn.Embedding(len(dset.attrs), word_vector_dim)
            self.obj_embedder = nn.Embedding(len(dset.objs), word_vector_dim)

            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

            if args.static_inp:
                for param in self.attr_embedder.parameters():
                    param.requires_grad = False
                for param in self.obj_embedder.parameters():
                    param.requires_grad = False

            self.layer_norm = nn.LayerNorm(word_vector_dim*2)
            self.MLP = nn.Sequential(
                nn.Linear(word_vector_dim*2,args.latent_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(args.latent_dim,args.emb_dim)
            )
        else:
            raise ValueError("Please enter the correct prototype learner name")

    def update_dict(self, wdict, row,col,data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)
    def init_embeddings(self, all_words):
        def get_compositional_embeddings(embeddings, pairs):
            # Getting compositional embeddings from base embeddings
            composition_embeds = []
            for (attr, obj) in pairs:
                attr_embed = embeddings[self.attr_to_idx[attr]]
                obj_embed = embeddings[self.obj_to_idx[obj]+self.num_attrs]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            print('Compositional Embeddings are ', composition_embeds.shape)
            return composition_embeds
        # init with word embeddings
        embeddings = load_word_embeddings(self.args.emb_init, all_words)
        composition_embeds = get_compositional_embeddings(embeddings, self.pairs)
        full_embeddings = torch.cat([embeddings, composition_embeds], dim=0)
        return full_embeddings
    def adj_from_pairs(self):

        def edges_from_pairs(pairs):
            weight_dict = {'data':[],'row':[],'col':[]}


            for i in range(self.displacement):
                self.update_dict(weight_dict,i,i,1.)

            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.attr_to_idx[attr], self.obj_to_idx[obj] + self.num_attrs

                self.update_dict(weight_dict, attr_idx, obj_idx, 1.)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.)

                node_id = idx + self.displacement
                self.update_dict(weight_dict,node_id,node_id,1.)

                self.update_dict(weight_dict, node_id, attr_idx, 1.)
                self.update_dict(weight_dict, node_id, obj_idx, 1.)


                self.update_dict(weight_dict, attr_idx, node_id, 1.)
                self.update_dict(weight_dict, obj_idx, node_id, 1.)

            return weight_dict

        edges = edges_from_pairs(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])),
                            shape=(len(self.pairs)+self.displacement, len(self.pairs)+self.displacement))
        return adj

    def forward(self, attrs, objs):
        if self.args.P_y == 'GCN':
            if self.training:
                output = self.gcn(self.embeddings)
                output = output[self.train_idx]
                return output
            else:
                output = self.gcn(self.embeddings)
                output = output[self.num_attrs + self.num_objs:self.num_attrs + self.num_objs +
                                                               self.num_pairs, :]
                return output

        elif self.args.P_y == 'MLP':
            attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
            inputs = torch.cat([attrs, objs], 1)
            output = self.MLP(inputs)
            output = F.normalize(output, dim=1)
            return output
        else:
            raise ValueError("Please enter the correct prototype learner name")


class _Vs_Vo(nn.Module):
    def __init__(self,dset, args):
        super(_Vs_Vo, self).__init__()
        layers2 = []
        self.dset = dset
        self.args = args
        self.args.latent_dim2 = self.args.latent_dim2.split(',')
        for dim in self.args.latent_dim2:
            dim = int(dim)
            layers2.append(dim)
        self.obj_img_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=2,
                                    dropout=self.args.dropout,
                                    norm=self.args.norm, layers=layers2)
        self.attr_img_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=False, num_layers=2,
                                     dropout=self.args.dropout,
                                     norm=self.args.norm, layers=layers2)
    def forward(self,img):
        attr_img = F.normalize(self.attr_img_embedder(img), dim=1)
        obj_img = F.normalize(self.obj_img_embedder(img), dim=1)
        return attr_img,obj_img

class _Ps_Po(nn.Module):
    def __init__(self, dset, args):
        super(_Ps_Po, self).__init__()
        self.dset = dset
        self.args = args

        '''get word vectors'''
        if args.emb_init == 'word2vec':
            word_vector_dim = 300
        elif args.emb_init == 'glove':
            word_vector_dim = 300
        elif args.emb_init == 'fasttext':
            word_vector_dim = 300
        else:
            word_vector_dim = 600

        self.attr_embedder = nn.Embedding(len(dset.attrs), word_vector_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), word_vector_dim)

        pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
        self.attr_embedder.weight.data.copy_(pretrained_weight)
        pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
        self.obj_embedder.weight.data.copy_(pretrained_weight)

        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        self.obj_projection = nn.Linear(word_vector_dim, args.emb_dim)
        self.attr_projection = nn.Linear(word_vector_dim, args.emb_dim)

    def forward(self,attrs,objs):
        objs_ = F.normalize(self.obj_projection(F.leaky_relu(self.obj_embedder(objs), inplace=True)), dim=1)
        attr_ = F.normalize(self.attr_projection(F.leaky_relu(self.attr_embedder(attrs), inplace=True)), dim=1)
        return attr_, objs_