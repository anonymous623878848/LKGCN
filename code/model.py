"""
create on 9th Nov, 2021
pytorch implementation of lightGCN
author: qinqin

model define
"""
import numpy as np
import torch as t
import world
from dataloader import BasicDataset
from torch import nn


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseBasicModel(BasicModel):
    def __init__(self):
        super(PairWiseBasicModel, self).__init__()

    def bpr_loss(self, users, pos, negs):
        """
        :param users: user list
        :param pos: positive items for corresponding user
        :param negs: negative items for corresponding user
        :return:
        (logloss, l2 loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 datasets: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = datasets.n_users
        self.num_items = datasets.m_items
        self.latent_dim = config["latent_dim_rec"]
        self.sig = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.user_emd = t.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.latent_dim
        )
        self.item_emd = t.nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim
        )
        print("use normal distribution N(0, 1) init for pureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.user_emd(users)
        items_emd = self.item_emd.weight
        scores = t.matmul(users_emb, items_emd.t())
        return self.sig(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.user_emd(users.long())
        pos_emb = self.item_emd(pos.long())
        neg_emb = self.item_emd(neg.long())
        pos_score = t.sum(users_emb * pos_emb, dim=1)
        neg_score = t.sum(users_emb * neg_emb, dim=1)
        # ???? why do you construct loss like this
        loss = t.mean(nn.functional.softplus(neg_score - pos_score))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users_emb = self.user_emd(users.long())
        items_emb = self.item_emd(items.long())
        scores = t.sum(users_emb * items_emb, dim=1)
        return self.sig(scores)


class PureEntity(BasicModel):
    def __init__(self,
                 config: dict,
                 datasets: BasicDataset):
        super(PureEntity, self).__init__()
        self.num_users = datasets.n_users
        self.num_items = datasets.m_items
        self.num_entities = datasets.n_others
        self.latent_dim = config["latent_dim_rec"]
        self.sig = nn.Sigmoid()
        self.__init_weight()
        self.UEnet, self.IEnet = datasets.getEntityNet()

    def getUserEmb(self):
        return t.matmul(self.UEnet, self.enti_emd.weight)

    def getItemEmb(self):
        return t.matmul(self.IEnet, self.enti_emd.weight)

    @property
    def user_emd(self):
        return self.getUserEmb()

    @property
    def item_emd(self):
        return self.getItemEmb()

    def __init_weight(self):
        self.enti_emd = t.nn.Embedding(
            num_embeddings=self.num_entities,
            embedding_dim=self.latent_dim
        )
        #         self.user_emd = t.nn.Embedding(
        #             num_embeddings=self.num_users,
        #             embedding_dim=self.latent_dim
        #         )
        #         self.item_emd = t.nn.Embedding(
        #             num_embeddings=self.num_items,
        #             embedding_dim=self.latent_dim
        #         )
        print("use normal distribution N(0, 1) init for pureEntity")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.user_emd[users]
        items_emd = self.item_emd
        scores = t.matmul(users_emb, items_emd.t())
        return self.sig(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.user_emd[users.long()]
        pos_emb = self.item_emd[pos.long()]
        neg_emb = self.item_emd[neg.long()]
        pos_score = t.sum(users_emb * pos_emb, dim=1)
        neg_score = t.sum(users_emb * neg_emb, dim=1)
        # ???? why do you construct loss like this
        loss = t.mean(nn.functional.softplus(neg_score - pos_score))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users_emb = self.user_emd[users.long()]
        items_emb = self.item_emd[items.long()]
        scores = t.sum(users_emb * items_emb, dim=1)
        return self.sig(scores)


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        # ###########
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lgn_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["a_split"]
        self.user_emds = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_emds = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config["pretrain"] == 0:
            nn.init.normal_(self.user_emds.weight, std=0.1)
            nn.init.normal_(self.item_emds.weight, std=0.1)
            world.cprint("use NORMAL distribution to init")
        else:
            self.user_emds.weight.data.copy_(t.from_numpy(self.config["user_emb"]))
            self.item_emds.weight.data.copy_(t.from_numpy(self.config["item_emb"]))
            world.cprint("use pretrain embedding")

        self.sig = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is ready to go(dropout: {self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def compute(self):
        """
        propagate methods for light GCN
        :return:
        """
        user_emds = self.user_emds.weight
        item_emds = self.item_emds.weight
        all_emds = t.cat([user_emds, item_emds])
        embs = [all_emds]
        if self.config["dropout"]:
            if self.training:
                print("dropping")
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(t.sparse.mm(g_dropped[f], all_emds))
                side_emb = t.cat(temp_emb, dim=0)
                all_emds = side_emb
            else:
                all_emds = t.sparse.mm(g_dropped, all_emds)

            embs.append(all_emds)
        embs = t.stack(embs, dim=1)
        lgn_out = t.mean(embs, dim=1)
        users, items = t.split(lgn_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.compute()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.sig(t.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]
        users_emb_ego = self.user_emds(users.long())
        pos_emb_ego = self.item_emds(pos_items.long())
        neg_emb_ego = self.item_emds(neg_items.long())
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (
            user_emb, pos_emb, neg_emb,
            userEmb0, posEmb0, negEmb0
        ) = self.getEmbedding(users, pos, neg)
        reg_loss = (1 / 2) * (
                userEmb0.norm(2).pow(2) +
                posEmb0.norm(2).pow(2) +
                negEmb0.norm(2).pow(2)
        ) / float(len(users))
        pos_scores = t.mul(user_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(user_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.compute()

        users_embs = all_users[users.long()]
        items_embs = all_items[items.long()]
        inner_pro = t.mul(users_embs, items_embs)
        gamma = t.sum(inner_pro, dim=1)
        return gamma


class LightKGGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightKGGCN, self).__init__()
        self.config = config
        # ###########
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_others = self.dataset.n_others
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lgn_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["a_split"]
        self.user_emds = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_emds = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.other_emds = nn.Embedding(num_embeddings=self.num_others, embedding_dim=self.latent_dim)
        if self.config["pretrain"] == 0:
            nn.init.normal_(self.user_emds.weight, std=0.1)
            nn.init.normal_(self.item_emds.weight, std=0.1)
            nn.init.normal_(self.other_emds.weight, std=0.1)
            world.cprint("use NORMAL distribution to init")
        else:
            self.user_emds.weight.data.copy_(t.from_numpy(self.config["user_emb"]))
            self.item_emds.weight.data.copy_(t.from_numpy(self.config["item_emb"]))
            self.other_emds.weiht.data.copy_(t.from_numpy(self.config["other_emb"]))
            world.cprint("use pretrain embedding")

        self.sig = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is ready to go(dropout: {self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def compute(self):
        """
        propagate methods for light GCN
        :return:
        """
        user_emds = self.user_emds.weight
        item_emds = self.item_emds.weight
        other_emds = self.other_emds.weight
        all_emds = t.cat([user_emds, item_emds, other_emds])
        embs = [all_emds]
        if self.config["dropout"]:
            if self.training:
                print("dropping")
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(t.sparse.mm(g_dropped[f], all_emds))
                side_emb = t.cat(temp_emb, dim=0)
                all_emds = side_emb
            else:
                all_emds = t.sparse.mm(g_dropped, all_emds)

            embs.append(all_emds)
        embs = t.stack(embs, dim=1)
        lgn_out = t.mean(embs, dim=1)
        users, items, others = t.split(lgn_out, [self.num_users, self.num_items, self.num_others])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.compute()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.sig(t.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]
        users_emb_ego = self.user_emds(users.long())
        pos_emb_ego = self.item_emds(pos_items.long())
        neg_emb_ego = self.item_emds(neg_items.long())
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (
            user_emb, pos_emb, neg_emb,
            userEmb0, posEmb0, negEmb0
        ) = self.getEmbedding(users, pos, neg)
        reg_loss = (1 / 2) * (
                userEmb0.norm(2).pow(2) +
                posEmb0.norm(2).pow(2) +
                negEmb0.norm(2).pow(2)
        ) / float(len(users))
        pos_scores = t.mul(user_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(user_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.compute()

        users_embs = all_users[users.long()]
        items_embs = all_items[items.long()]
        inner_pro = t.mul(users_embs, items_embs)
        gamma = t.sum(inner_pro, dim=1)
        return gamma


class LightGCNMulti(PairWiseBasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCNMulti, self).__init__()
        self.config = config
        # ###########
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_others = self.dataset.n_others
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lgn_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["a_split"]
        self.user_emds = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_emds = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.other_emds = nn.Embedding(num_embeddings=self.num_others, embedding_dim=self.latent_dim)
        if self.config["pretrain"] == 0:
            nn.init.normal_(self.user_emds.weight, std=0.1)
            nn.init.normal_(self.item_emds.weight, std=0.1)
            nn.init.normal_(self.other_emds.weight, std=0.1)
            world.cprint("use NORMAL distribution to init")
        else:
            self.user_emds.weight.data.copy_(t.from_numpy(self.config["user_emb"]))
            self.item_emds.weight.data.copy_(t.from_numpy(self.config["item_emb"]))
            self.other_emds.weiht.data.copy_(t.from_numpy(self.config["other_emb"]))
            world.cprint("use pretrain embedding")

        self.sig = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is ready to go(dropout: {self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def compute(self):
        """
        propagate methods for light GCN
        :return:
        """
        user_emds = self.user_emds.weight
        item_emds = self.item_emds.weight
        other_emds = self.other_emds.weight
        all_emds = t.cat([user_emds, item_emds, other_emds])
        embs = [all_emds]
        if self.config["dropout"]:
            if self.training:
                print("dropping")
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(t.sparse.mm(g_dropped[f], all_emds))
                side_emb = t.cat(temp_emb, dim=0)
                all_emds = side_emb
            else:
                all_emds = t.sparse.mm(g_dropped, all_emds)

            embs.append(all_emds)
        embs = t.stack(embs, dim=1)
        lgn_out = t.mean(embs, dim=1)
        users, items, others = t.split(lgn_out, [self.num_users, self.num_items, self.num_others])
        return users, items, others

    def getUsersRating(self, users):
        all_users, all_items, _ = self.compute()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.sig(t.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items, usersO, pos_others, neg_others):
        all_users, all_items, all_others = self.compute()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]
        users_emb_ego = self.user_emds(users.long())
        pos_emb_ego = self.item_emds(pos_items.long())
        neg_emb_ego = self.item_emds(neg_items.long())

        usersOEmbs = all_users[usersO.long()]
        pos_Oemb = all_others[pos_others.long()]
        neg_Oemb = all_others[neg_others.long()]
        usersOEmbes_ego = self.user_emds(usersO.long())
        posOemb_ego = self.other_emds(pos_others.long())
        negOemb_ego = self.other_emds(neg_others.long())

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, usersOEmbs, pos_Oemb, neg_Oemb, usersOEmbes_ego, posOemb_ego, negOemb_ego

    def bpr_loss(self, users, pos, neg, usersO, posO, negO):
        (
            user_emb, pos_emb, neg_emb,
            userEmb0, posEmb0, negEmb0,
            userO_emb, posO_emb, negO_emb,
            userOEmb0, posOEmb0, negOEmb0,
        ) = self.getEmbedding(users, pos, neg, usersO, posO, negO)
        reg_loss0 = (1 / 2) * (
                userEmb0.norm(2).pow(2) +
                posEmb0.norm(2).pow(2) +
                negEmb0.norm(2).pow(2)
        ) / float(len(users))
        reg_loss1 = (1 / 2) * (
                userOEmb0.norm(2).pow(2) +
                posOEmb0.norm(2).pow(2) +
                negOEmb0.norm(2).pow(2)
        ) / float(len(usersO))
        pos_scores = t.mul(user_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(user_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        posO_scores = t.mul(userO_emb, posO_emb)
        posO_scores = t.sum(posO_scores, dim=1)
        negO_scores = t.mul(userO_emb, negO_emb)
        negO_scores = t.sum(negO_scores, dim=1)

        loss0 = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))
        loss1 = t.mean(t.nn.functional.softplus(negO_scores - posO_scores))
        return loss0, reg_loss0, loss1, reg_loss1

    def forward(self, users, items):
        all_users, all_items = self.compute()

        users_embs = all_users[users.long()]
        items_embs = all_items[items.long()]
        inner_pro = t.mul(users_embs, items_embs)
        gamma = t.sum(inner_pro, dim=1)
        return gamma


class LightGCNMultiEntity(PairWiseBasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCNMultiEntity, self).__init__()
        self.config = config
        # ###########
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()
        self.UEnet, self.IEnet = dataset.getEntityNet()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_others = self.dataset.n_others
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lgn_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["a_split"]
        self.user_emds = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_emds = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.other_emds = nn.Embedding(num_embeddings=self.num_others, embedding_dim=self.latent_dim)
        if self.config["pretrain"] == 0:
            nn.init.normal_(self.user_emds.weight, std=0.1)
            nn.init.normal_(self.item_emds.weight, std=0.1)
            nn.init.normal_(self.other_emds.weight, std=0.1)
            world.cprint("use NORMAL distribution to init")
        else:
            self.user_emds.weight.data.copy_(t.from_numpy(self.config["user_emb"]))
            self.item_emds.weight.data.copy_(t.from_numpy(self.config["item_emb"]))
            self.other_emds.weiht.data.copy_(t.from_numpy(self.config["other_emb"]))
            world.cprint("use pretrain embedding")

        self.sig = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is ready to go(dropout: {self.config['dropout']})")

    def getUserEmb(self):
        return t.matmul(self.UEnet, self.other_emds.weight)

    def getItemEmb(self):
        return t.matmul(self.IEnet, self.other_emds.weight)

    @property
    def userO_emd(self):
        return self.getUserEmb()

    @property
    def itemO_emd(self):
        return self.getItemEmb()

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def compute(self):
        """
        propagate methods for light GCN
        :return:
        """
        user_emds = self.user_emds.weight
        item_emds = self.item_emds.weight
        other_emds = self.other_emds.weight
        all_emds = t.cat([user_emds, item_emds, other_emds])
        embs = [all_emds]
        if self.config["dropout"]:
            if self.training:
                print("dropping")
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(t.sparse.mm(g_dropped[f], all_emds))
                side_emb = t.cat(temp_emb, dim=0)
                all_emds = side_emb
            else:
                all_emds = t.sparse.mm(g_dropped, all_emds)

            embs.append(all_emds)
        embs = t.stack(embs, dim=1)
        lgn_out = t.mean(embs, dim=1)
        users, items, others = t.split(lgn_out, [self.num_users, self.num_items, self.num_others])
        return users, items, others

    def getUsersRating(self, users):
        all_users, all_items, _ = self.compute()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.sig(t.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items, usersO, pos_others, neg_others):
        all_users, all_items, all_others = self.compute()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]
        users_emb_ego = self.user_emds(users.long())
        pos_emb_ego = self.item_emds(pos_items.long())
        neg_emb_ego = self.item_emds(neg_items.long())

        usersOEmbs = all_users[usersO.long()]
        pos_Oemb = all_others[pos_others.long()]
        neg_Oemb = all_others[neg_others.long()]
        usersOEmbes_ego = self.user_emds(usersO.long())
        posOemb_ego = self.other_emds(pos_others.long())
        negOemb_ego = self.other_emds(neg_others.long())

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, usersOEmbs, pos_Oemb, neg_Oemb, usersOEmbes_ego, posOemb_ego, negOemb_ego

    def bpr_loss(self, users, pos, neg, usersO, posO, negO):
        (
            user_emb, pos_emb, neg_emb,
            userEmb0, posEmb0, negEmb0,
            userO_emb, posO_emb, negO_emb,
            userOEmb0, posOEmb0, negOEmb0,
        ) = self.getEmbedding(users, pos, neg, usersO, posO, negO)
        reg_loss0 = (1 / 2) * (
                userEmb0.norm(2).pow(2) +
                posEmb0.norm(2).pow(2) +
                negEmb0.norm(2).pow(2)
        ) / float(len(users))
        reg_loss1 = (1 / 2) * (
                userOEmb0.norm(2).pow(2) +
                posOEmb0.norm(2).pow(2) +
                negOEmb0.norm(2).pow(2)
        ) / float(len(usersO))
        pos_scores = t.mul(user_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(user_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        posO_scores = t.mul(userO_emb, posO_emb)
        posO_scores = t.sum(posO_scores, dim=1)
        negO_scores = t.mul(userO_emb, negO_emb)
        negO_scores = t.sum(negO_scores, dim=1)

        # the third loss which constrain u-I-E relation
        #         uo_emb = self.userO_emd[users.long()]
        posIo_emb = self.itemO_emd[pos.long()]
        negIo_emb = self.itemO_emd[neg.long()]

        pos_scores_uo = t.mul(user_emb, posIo_emb)
        pos_scores_uo = t.sum(pos_scores_uo, dim=1)
        neg_scores_uo = t.mul(user_emb, negIo_emb)
        neg_scores_uo = t.sum(neg_scores_uo, dim=1)

        loss0 = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))
        loss1 = t.mean(t.nn.functional.softplus(negO_scores - posO_scores))
        loss3 = t.mean(t.nn.functional.softplus(neg_scores_uo - pos_scores_uo))

        reg_loss3 = (1 / 2) * (
                userEmb0.norm(2).pow(2) +
                posIo_emb.norm(2).pow(2) +
                negIo_emb.norm(2).pow(2)
        ) / float(len(users))
        return loss0, reg_loss0, loss1, reg_loss1, loss3, reg_loss3

    def forward(self, users, items):
        all_users, all_items = self.compute()

        users_embs = all_users[users.long()]
        items_embs = all_items[items.long()]
        inner_pro = t.mul(users_embs, items_embs)
        gamma = t.sum(inner_pro, dim=1)
        return gamma


class LightGCNMultiAtt(PairWiseBasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCNMultiAtt, self).__init__()
        self.config = config
        # ###########
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()
        self.UEnet, self.IEnet = dataset.getEntityNet()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_others = self.dataset.n_others
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lgn_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["a_split"]
        self.user_emds = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_emds = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.other_emds = nn.Embedding(num_embeddings=self.num_others, embedding_dim=self.latent_dim)
        # for ue and ui consistency
        self.fc = nn.Linear(self.latent_dim, 1)
        if self.config["pretrain"] == 0:
            nn.init.normal_(self.user_emds.weight, std=0.1)
            nn.init.normal_(self.item_emds.weight, std=0.1)
            nn.init.normal_(self.other_emds.weight, std=0.1)
            world.cprint("use NORMAL distribution to init")
        else:
            self.user_emds.weight.data.copy_(t.from_numpy(self.config["user_emb"]))
            self.item_emds.weight.data.copy_(t.from_numpy(self.config["item_emb"]))
            self.other_emds.weiht.data.copy_(t.from_numpy(self.config["other_emb"]))
            world.cprint("use pretrain embedding")

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is ready to go(dropout: {self.config['dropout']})")

    def getUserEmb(self):
        return t.matmul(self.UEnet, self.other_emds.weight)

    def getItemEmb(self):
        return t.matmul(self.IEnet, self.other_emds.weight)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def compute(self):
        """
        propagate methods for light GCN
        :return:
        """
        user_emds = self.user_emds.weight
        item_emds = self.item_emds.weight
        other_emds = self.other_emds.weight
        all_emds = t.cat([user_emds, item_emds, other_emds])
        embs = [all_emds]
        if self.config["dropout"]:
            if self.training:
                print("dropping")
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(t.sparse.mm(g_dropped[f], all_emds))
                side_emb = t.cat(temp_emb, dim=0)
                all_emds = side_emb
            else:
                all_emds = t.sparse.mm(g_dropped, all_emds)

            embs.append(all_emds)
        embs = t.stack(embs, dim=1)
        lgn_out = t.mean(embs, dim=1)
        users, items, others = t.split(lgn_out, [self.num_users, self.num_items, self.num_others])
        return users, items, others

    def getUsersRating(self, users):
        all_users, all_items, _ = self.compute()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.sig(t.matmul(users_emb, items_emb.t()))
        return rating

    ####################
    def getEmbedding(self, users, pos_items, neg_items, usersO, pos_others, neg_others, itemEntitiesPos,
                     itemEntitiesNeg):
        all_users, all_items, all_others = self.compute()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]
        users_emb_ego = self.user_emds(users.long())
        pos_emb_ego = self.item_emds(pos_items.long())
        neg_emb_ego = self.item_emds(neg_items.long())

        usersOEmbs = all_users[usersO.long()]
        pos_Oemb = all_others[pos_others.long()]
        neg_Oemb = all_others[neg_others.long()]
        usersOEmbes_ego = self.user_emds(usersO.long())
        posOemb_ego = self.other_emds(pos_others.long())
        negOemb_ego = self.other_emds(neg_others.long())

        # get positive user-entities attention enbedding
        posEntiTensor_hot = self.onehot_map(itemEntitiesPos)
        posEntiTensor = posEntiTensor_hot.unsqueeze(dim=2) * all_others
        tem_userEmb = users_emb.unsqueeze(dim=1)
        posEntiTensor_T = posEntiTensor.transpose(1, 2)
        tem_userEmb = tem_userEmb.expand(posEntiTensor_T.shape[0], -1, -1)
        posAttweight = t.bmm(tem_userEmb, posEntiTensor_T)
        posAttweight = self.relu(posAttweight.squeeze(dim=1))
        mask = posAttweight != 0
        # I only need to consider non-zero entry value
        posAttweight = self.masked_softmax(posAttweight, mask)
        posAttweight_emb = t.matmul(posAttweight, all_others)

        # get negative user-entities attention enbedding
        negEntiTensor_hot = self.onehot_map(itemEntitiesNeg)
        negEntiTensor = negEntiTensor_hot.unsqueeze(dim=2) * all_others
        tem_userEmb = users_emb.unsqueeze(dim=1)
        negEntiTensor_T = negEntiTensor.transpose(1, 2)
        tem_userEmb = tem_userEmb.expand(negEntiTensor_T.shape[0], -1, -1)
        negAttweight = t.bmm(tem_userEmb, negEntiTensor_T)
        negAttweight = self.relu(negAttweight.squeeze(dim=1))
        mask = negAttweight != 0
        # I only need to consider non-zero entry value
        negAttweight = self.masked_softmax(negAttweight, mask)
        negAttweight_emb = t.matmul(negAttweight, all_others)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, usersOEmbs, pos_Oemb, neg_Oemb, usersOEmbes_ego, posOemb_ego, negOemb_ego, posAttweight_emb, negAttweight_emb

    def onehot_map(self, e):
        onehot_entities = np.zeros((len(e), self.num_others))
        for row, col_index in enumerate(e):
            onehot_entities[row, col_index] = 1
        entiTensor_hot = t.Tensor(onehot_entities).long().to(world.device)
        return entiTensor_hot

    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):
        exps = t.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)

    def bpr_loss(self, users, pos, neg, usersO, posO, negO, itemEntitiesPos, itemEntitiesNeg):
        (
            user_emb, pos_emb, neg_emb,
            userEmb0, posEmb0, negEmb0,
            userO_emb, posO_emb, negO_emb,
            userOEmb0, posOEmb0, negOEmb0,
            posAttweightEmb, negAttweightEmb
        ) = self.getEmbedding(users, pos, neg, usersO, posO, negO, itemEntitiesPos, itemEntitiesNeg)
        reg_loss0 = (1 / 2) * (
                userEmb0.norm(2).pow(2) +
                posEmb0.norm(2).pow(2) +
                negEmb0.norm(2).pow(2)
        ) / float(len(users))
        reg_loss1 = (1 / 2) * (
                userOEmb0.norm(2).pow(2) +
                posOEmb0.norm(2).pow(2) +
                negOEmb0.norm(2).pow(2)
        ) / float(len(usersO))
        pos_scores = t.mul(user_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(user_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        posO_scores = t.mul(userO_emb, posO_emb)
        posO_scores = t.sum(posO_scores, dim=1)
        negO_scores = t.mul(userO_emb, negO_emb)
        negO_scores = t.sum(negO_scores, dim=1)

        # the third loss which constrain u-I-E relation
        pos_entity_scores = self.fc(posAttweightEmb)
        pos_entity_scores = t.sum(pos_entity_scores, dim=1)
        neg_entity_scores = self.fc(negAttweightEmb)
        neg_entity_scores = t.sum(neg_entity_scores, dim=1)

        loss0 = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))
        loss1 = t.mean(t.nn.functional.softplus(negO_scores - posO_scores))
        loss3 = t.mean(t.nn.functional.softplus(neg_entity_scores - pos_entity_scores))

        reg_loss3 = (1 / 2) * (
                userEmb0.norm(2).pow(2) +
                posAttweightEmb.norm(2).pow(2) +
                negAttweightEmb.norm(2).pow(2)
        ) / float(len(users))
        return loss0, reg_loss0, loss1, reg_loss1, loss3, reg_loss3

    def forward(self, users, items):
        all_users, all_items = self.compute()

        users_embs = all_users[users.long()]
        items_embs = all_items[items.long()]
        inner_pro = t.mul(users_embs, items_embs)
        gamma = t.sum(inner_pro, dim=1)
        return gamma
