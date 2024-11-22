import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


# Semantic attention in the metapath-based aggregation (the same as that in the HAN)
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):  #z:Tensor(2646,4,128)
        """
        Shape of z: (N, M , D*K)
        N: number of nodes
        M: number of metapath patterns
        D: hidden_size
        K: number of heads
        """
        w = self.project(z).mean(0)  # (M, 1)   #tensor([[-0.0163],
                                                        # [-0.0163],
                                                        # [-0.0163],
                                                        # [-0.0163]], grad_fn=<MeanBackward1>)
        beta = torch.softmax(w, dim=0)  # (M, 1)      tensor([[0.2500],
                                                            # [0.2500],
                                                            # [0.2500],
                                                            # [0.2500]], grad_fn=<SoftmaxBackward0>)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


# Metapath-based aggregation (the same as the HANLayer)
class HANLayer(nn.Module):
    def __init__(
        self, meta_path_patterns, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_path_patterns)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_path_patterns = list(
            tuple(meta_path_pattern) for meta_path_pattern in meta_path_patterns
        )

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        # obtain metapath reachable graph
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path_pattern in self.meta_path_patterns:
                self._cached_coalesced_graph[
                    meta_path_pattern
                ] = dgl.metapath_reachable_graph(g, meta_path_pattern)

        for i, meta_path_pattern in enumerate(self.meta_path_patterns):
            new_g = self._cached_coalesced_graph[meta_path_pattern]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)          # semantic_embeddings={Tensor(2646,4,128)}


# Relational neighbor aggregation
class RelationalAGG(nn.Module):
    def __init__(self, g, in_size, out_size, dropout=0.1):
        super(RelationalAGG, self).__init__()
        self.in_size = in_size             #128
        self.out_size = out_size            #128

        # Transform weights for different types of edges
        self.W_T = nn.ModuleDict(                   #ModuleDict(
                                                                #   (bi): Linear(in_features=128, out_features=128, bias=False)
                                                                #   (ci): Linear(in_features=128, out_features=128, bias=False)
                                                                #   (ib): Linear(in_features=128, out_features=128, bias=False)
                                                                #   (ic): Linear(in_features=128, out_features=128, bias=False)
                                                                #   (iu): Linear(in_features=128, out_features=128, bias=False)
                                                                #   (iv): Linear(in_features=128, out_features=128, bias=False)
                                                                #   (ui): Linear(in_features=128, out_features=128, bias=False)
                                                                #   (vi): Linear(in_features=128, out_features=128, bias=False)
                                                                # )
            {
                name: nn.Linear(in_size, out_size, bias=False)
                for name in g.etypes
            }
        )

        # Attention weights for different types of edges   计算每一条边的权重，对于这个异构图，每条边对于一个节点的嵌入有多大的帮助
        self.W_A = nn.ModuleDict(                      #  (W_A): ModuleDict(
                                                                          #   (bi): Linear(in_features=128, out_features=1, bias=False)
                                                                          #   (ci): Linear(in_features=128, out_features=1, bias=False)
                                                                          #   (ib): Linear(in_features=128, out_features=1, bias=False)
                                                                          #   (ic): Linear(in_features=128, out_features=1, bias=False)
                                                                          #   (iu): Linear(in_features=128, out_features=1, bias=False)
                                                                          #   (iv): Linear(in_features=128, out_features=1, bias=False)
                                                                          #   (ui): Linear(in_features=128, out_features=1, bias=False)
                                                                          #   (vi): Linear(in_features=128, out_features=1, bias=False)
                                                                          # )
            {name: nn.Linear(out_size, 1, bias=False) for name in g.etypes}
        )

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)               #  (layernorm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat_dict):    #ParameterDict(
                                                        #     (brand): Parameter containing: [torch.FloatTensor of size 5x128]
                                                        #     (category): Parameter containing: [torch.FloatTensor of size 2x128]
                                                        #     (item): Parameter containing: [torch.FloatTensor of size 2646x128]
                                                        #     (user): Parameter containing: [torch.FloatTensor of size 6157x128]
                                                        #     (view): Parameter containing: [torch.FloatTensor of size 13x128]
                                                        # )
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:           # canonical_etypes： [('brand', 'bi', 'item'), ('category', 'ci', 'item'), ('item', 'ib', 'brand'), ('item', 'ic', 'category'), ('item', 'iu', 'user'), ('item', 'iv', 'view'), ('user', 'ui', 'item'), ('view', 'vi', 'item')]
            g.nodes[dsttype].data["h"] = feat_dict[
                dsttype
            ]  # nodes' original feature
            g.nodes[srctype].data["h"] = feat_dict[srctype]
            g.nodes[srctype].data["t_h"] = self.W_T[etype](    #W_T相当于经过了一个特定的神经网络  in：128 out:128
                feat_dict[srctype]
            )  # src nodes' transformed feature   原节点transformd后的特征

            # compute the attention numerator (exp)   计算每条边的一个权值
            g.apply_edges(fn.u_mul_v("t_h", "h", "x"), etype=etype)     #通过在src和dst特征之间执行逐元素乘法来计算边缘上的消息
            g.edges[etype].data["x"] = torch.exp(
                self.W_A[etype](g.edges[etype].data["x"])
            )

            # first update to compute the attention denominator (\sum exp)
            funcs[etype] = (fn.copy_e("x", "m"), fn.sum("m", "att"))     #copy_e 可使用边特征计算消息。  按sum来汇总消息。
        g.multi_update_all(funcs, "sum")

        funcs = {}               #{'bi': (<dgl.function.message.BinaryMessageFunction object at 0x00000285903BE898>, <dgl.function.reducer.SimpleReduceFunction object at 0x000002858F9FB438>), 'ci': (<dgl.function.message.BinaryMessageFunction object at 0x000002858ACF2B70>, <dgl.function.reducer.SimpleReduceFunction object at 0x0000028590245208>), 'ib': (<dgl.function.message.BinaryMessageFunction object at 0x0000028590251EB8>, <dgl.function.reducer.SimpleReduceFunction object at 0x0000028586EBEBE0>), 'ic': (<dgl.function.message.BinaryMessageFunction object at 0x000002859044FBE0>, <dgl.function.reducer.SimpleReduceFunction object at 0x000002858F9E24A8>), 'iu': (<dgl.function.message.BinaryMessageFunction object at 0x0000028586EBEBA8>, <dgl.function.reducer.SimpleReduceFunction object at 0x0000028590445588>), 'iv': (<dgl.function.message.BinaryMessageFunction object at 0x0000028591B694E0>, <dgl.function.reducer.SimpleReduceFunction object at 0x000002858ACF2CF8>), 'ui': (<dgl.function.message.BinaryMessageFunction object at 0x000002858F9FB400>, <dgl.function.reducer.SimpleReduceFunction object at 0x000002858F1B3E80>), 'vi': (<dgl.function.message.BinaryMessageFunction object at 0x000002858F9E2748>, <dgl.function.reducer.SimpleReduceFunction object at 0x0000028591B69A20>)}
        for srctype, etype, dsttype in g.canonical_etypes:
            g.apply_edges(
                fn.e_div_v("x", "att", "att"), etype=etype    #在edge和dst的特征之间执行逐元素除法来计算边缘上的消息
            )  # compute attention weights (numerator/denominator)
            funcs[etype] = (
                fn.u_mul_e("h", "att", "m"),      #在src和edge特征之间执行逐元素乘法来计算边缘上的消息
                fn.sum("m", "h"),
            )  # \sum(h0*att) -> h1
        # second update to obtain h1
        g.multi_update_all(funcs, "sum")

        # apply activation, layernorm, and dropout
        feat_dict = {}
        for ntype in g.ntypes:
            feat_dict[ntype] = self.dropout(
                self.layernorm(F.relu_(g.nodes[ntype].data["h"]))    #(layernorm): LayerNorm((128,), eps=1e-05, elementwise_affine=True) 再经过一个layernorm层   归一化
            )  # apply activation, layernorm, and dropout

        return feat_dict   #每个节点


class TAHIN(nn.Module):
    '''
    TAHIN(
  (feature_dict): ParameterDict(
      (brand): Parameter containing: [torch.FloatTensor of size 5x128]
      (category): Parameter containing: [torch.FloatTensor of size 2x128]
      (item): Parameter containing: [torch.FloatTensor of size 2646x128]
      (user): Parameter containing: [torch.FloatTensor of size 6157x128]
      (view): Parameter containing: [torch.FloatTensor of size 13x128]
  )
  (RelationalAGG): RelationalAGG(
    (W_T): ModuleDict(
      (bi): Linear(in_features=128, out_features=128, bias=False)
      (ci): Linear(in_features=128, out_features=128, bias=False)
      (ib): Linear(in_features=128, out_features=128, bias=False)
      (ic): Linear(in_features=128, out_features=128, bias=False)
      (iu): Linear(in_features=128, out_features=128, bias=False)
      (iv): Linear(in_features=128, out_features=128, bias=False)
      (ui): Linear(in_features=128, out_features=128, bias=False)
      (vi): Linear(in_features=128, out_features=128, bias=False)
    )
    (W_A): ModuleDict(
      (bi): Linear(in_features=128, out_features=1, bias=False)
      (ci): Linear(in_features=128, out_features=1, bias=False)
      (ib): Linear(in_features=128, out_features=1, bias=False)
      (ic): Linear(in_features=128, out_features=1, bias=False)
      (iu): Linear(in_features=128, out_features=1, bias=False)
      (iv): Linear(in_features=128, out_features=1, bias=False)
      (ui): Linear(in_features=128, out_features=1, bias=False)
      (vi): Linear(in_features=128, out_features=1, bias=False)
    )
    (layernorm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (hans): ModuleDict(
    (user): HANLayer(
      (gat_layers): ModuleList(
        (0): GATConv(
          (fc): Linear(in_features=128, out_features=128, bias=False)
          (feat_drop): Dropout(p=0.1, inplace=False)
          (attn_drop): Dropout(p=0.1, inplace=False)
          (leaky_relu): LeakyReLU(negative_slope=0.2)
        )
      )
      (semantic_attention): SemanticAttention(
        (project): Sequential(
          (0): Linear(in_features=128, out_features=128, bias=True)
          (1): Tanh()
          (2): Linear(in_features=128, out_features=1, bias=False)
        )
      )
    )
    (item): HANLayer(
      (gat_layers): ModuleList(
        (0): GATConv(
          (fc): Linear(in_features=128, out_features=128, bias=False)
          (feat_drop): Dropout(p=0.1, inplace=False)
          (attn_drop): Dropout(p=0.1, inplace=False)
          (leaky_relu): LeakyReLU(negative_slope=0.2)
        )
        (1): GATConv(
          (fc): Linear(in_features=128, out_features=128, bias=False)
          (feat_drop): Dropout(p=0.1, inplace=False)
          (attn_drop): Dropout(p=0.1, inplace=False)
          (leaky_relu): LeakyReLU(negative_slope=0.2)
        )
        (2): GATConv(
          (fc): Linear(in_features=128, out_features=128, bias=False)
          (feat_drop): Dropout(p=0.1, inplace=False)
          (attn_drop): Dropout(p=0.1, inplace=False)
          (leaky_relu): LeakyReLU(negative_slope=0.2)
        )
        (3): GATConv(
          (fc): Linear(in_features=128, out_features=128, bias=False)
          (feat_drop): Dropout(p=0.1, inplace=False)
          (attn_drop): Dropout(p=0.1, inplace=False)
          (leaky_relu): LeakyReLU(negative_slope=0.2)
        )
      )
      (semantic_attention): SemanticAttention(
        (project): Sequential(
          (0): Linear(in_features=128, out_features=128, bias=True)
          (1): Tanh()
          (2): Linear(in_features=128, out_features=1, bias=False)
        )
      )
    )
  )
  (user_layer1): Linear(in_features=256, out_features=128, bias=True)
  (user_layer2): Linear(in_features=256, out_features=128, bias=True)
  (item_layer1): Linear(in_features=256, out_features=128, bias=True)
  (item_layer2): Linear(in_features=256, out_features=128, bias=True)
  (layernorm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (pred): Linear(in_features=128, out_features=128, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc): Linear(in_features=128, out_features=1, bias=True)
)
    '''
    def __init__(
        self, g, meta_path_patterns, in_size, out_size, num_heads, dropout   #Graph(num_nodes={'brand': 5, 'category': 2, 'item': 2646, 'user': 6157, 'view': 13},
                                                                              # num_edges={('brand', 'bi', 'item'): 5, ('category', 'ci', 'item'): 9, ('item', 'ib', 'brand'): 5, ('item', 'ic', 'category'): 9, ('item', 'iu', 'user'): 104, ('item', 'iv', 'view'): 13, ('user', 'ui', 'item'): 104, ('view', 'vi', 'item'): 13},
                                                                              # metagraph=[('brand', 'item', 'bi'), ('item', 'brand', 'ib'), ('item', 'category', 'ic'), ('item', 'user', 'iu'), ('item', 'view', 'iv'), ('category', 'item', 'ci'), ('user', 'item', 'ui'), ('view', 'item', 'vi')])
    ):
        super(TAHIN, self).__init__()

        # embeddings for different types of nodes, h0
        self.initializer = nn.init.xavier_uniform_     #参数初始化方法，用于初始化神经网络的权重
        self.feature_dict = nn.ParameterDict(         #ParameterDict(
                                                                    # (brand): Parameter containing: [torch.FloatTensor of size 5x128]
                                                                    # (category): Parameter containing: [torch.FloatTensor of size 2x128]
                                                                    # (item): Parameter containing: [torch.FloatTensor of size 2646x128]
                                                                    # (user): Parameter containing: [torch.FloatTensor of size 6157x128]
                                                                    # (view): Parameter containing: [torch.FloatTensor of size 13x128]
                                                                    # )
            {
                ntype: nn.Parameter(
                    self.initializer(torch.empty(g.num_nodes(ntype), in_size))
                )
                for ntype in g.ntypes
            }
        )

        # relational neighbor aggregation, this produces h1
        self.RelationalAGG = RelationalAGG(g, in_size, out_size)

        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict(
            {
                key: HANLayer(value, in_size, out_size, num_heads, dropout)
                for key, value in self.meta_path_patterns.items()
            }
        )

        # layers to combine h0, h1, and h2
        # used to update node embeddings
        self.user_layer1 = nn.Linear(
            (num_heads + 1) * out_size, out_size, bias=True
        )
        self.user_layer2 = nn.Linear(2 * out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(
            (num_heads + 1) * out_size, out_size, bias=True
        )
        self.item_layer2 = nn.Linear(2 * out_size, out_size, bias=True)

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

        # network to score the node pairs
        self.pred = nn.Linear(out_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_size, 1)

    def forward(self, g, user_key, item_key,user_idx,item_idx):         #用户节点的编号： user_idx： tensor([5812, 5362, 4037, 6015,  948, 5484, 1675, 3450,  490,  820,  367,  846,
                                                                                #  387, 4032, 4803, 2709, 2245, 5134,  581, 3974, 4044, 5393,   38,    1,
                                                                                #  176, 5845,  653, 4027,  516, 1498,  503,  534, 1727, 1231, 3680, 5524,
                                                                                # 2606,  286, 2510, 4276, 1016, 5274, 1600, 4283, 2013,  799,  633, 2819,
                                                                                # 1449, 2014,   40, 3991, 3962, 5896, 1042, 3504, 6029, 1392, 4621, 3013,
                                                                                #  437, 6089,    1, 3148,  214,  403, 5563, 5122, 3962,  781, 3960, 4280,
                                                                                #   97,  209, 5382,  614,  237, 1239,  175, 2081, 1922,  237, 5541,   44,
                                                                                # 5764,  214, 2212,  155, 3205, 5145,  217, 4321,  124,  388,  337, 5227,
                                                                                # 5889,    1, 4256, 5905,  414,  773, 4158, 3985, 2671, 4992,  283,  835,
                                                                                #  404, 5401, 2508, 3803, 2977, 1124, 5863,  346, 5517,  965, 2061, 2076,
                                                                                #  380, 2024, 2897,  512, 4263,  740, 2780, 4744])   128个batch中的用户编号
                                                                        #物品节点的编号： item_idx： tensor([ 894,   16,  550, 2442, 1649, 1347, 1213, 1170, 2182, 1247,  359, 2070,
                                                                                # 1170,  119, 1814, 2616,  681, 1746, 1769, 1396, 2616, 1769, 1769, 1170,
                                                                                # 2070, 1084, 1170,   54, 1170, 1170,  677, 2399, 2616, 2070, 2616, 2070,
                                                                                # 2645, 2070, 1170, 2645, 1170,  469, 2399, 1046, 2399,  734, 1170, 2138,
                                                                                # 1170, 2182, 1769, 2070, 2616, 2399, 2399, 2399, 1170, 2616, 1769, 1170,
                                                                                # 1170, 2565, 2399,  151, 2070, 1170, 2070, 1456, 1695, 1170, 2616, 1769,
                                                                                # 1170, 1170, 2231, 2070, 1769,  552, 1170, 1170, 1170, 2070, 1161, 1170,
                                                                                # 2339, 1769, 1170, 1170, 1170,  308, 2070,   58, 2556, 2616, 1613, 2411,
                                                                                #  578, 2070, 2070, 1230, 1170, 2181, 1769, 2243, 2011,  532, 2616, 2070,
                                                                                # 1170, 2645, 1242,  481, 2616, 1170, 1300, 1769, 2376,  828, 2163, 1682,
                                                                                # 1684, 1891, 2349, 2333,  286, 1769, 1461, 1170])
                                                                        #h1 = {dict:5}  'brand'={Tensor(5,128)}  'category'={Tensor(2,128) 'item'={Tensor(2646,128) 'user'={Tensor(6157,128) 'view'={Tensor(13,128)}

        # relational neighbor aggregation, h1
        h1 = self.RelationalAGG(g, self.feature_dict)

        # metapath-based aggregation, h2
        h2 = {}
        '''
        h2:{dict：2}     user{Tensor(6157,128)}  item{Tensor(2646,128)}
        {'user': tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0013, -0.0005,  0.0018,  ..., -0.0009,  0.0015, -0.0012],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       grad_fn=<SumBackward1>), 'item': tensor([[ 0.0050, -0.0058, -0.0036,  ...,  0.0042, -0.0203,  0.0157],
        [ 0.0126, -0.0110,  0.0085,  ...,  0.0020,  0.0030,  0.0199],
        [ 0.0021, -0.0094, -0.0045,  ..., -0.0149, -0.0162,  0.0151],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0182, -0.0020,  0.0086,  ...,  0.0033,  0.0042, -0.0211]],
       grad_fn=<SumBackward1>)}
        '''
        for key in self.meta_path_patterns.keys():
            h2[key] = self.hans[key](g, self.feature_dict[key])

        # update node embeddings
        user_emb = torch.cat((h1[user_key], h2[user_key]), 1)            #user_emb: Tensor(6157,256)
        item_emb = torch.cat((h1[item_key], h2[item_key]), 1)              #item_emb: Tensor(2646,256)
        user_emb = self.user_layer1(user_emb)                            #user_emb: Tensor(6157,128)
        item_emb = self.item_layer1(item_emb)                          #item_emb: Tensor(2646,128)
        user_emb = self.user_layer2(    #user_emb: Tensor(6157,128)
            torch.cat((user_emb, self.feature_dict[user_key]), 1)
        )
        item_emb = self.item_layer2(         #item_emb: Tensor(2646,128)
            torch.cat((item_emb, self.feature_dict[item_key]), 1)
        )

        # Relu
        user_emb = F.relu_(user_emb)            #user_emb: Tensor(6157,128)
        item_emb = F.relu_(item_emb)            #item_emb: Tensor(2646,128)

        # layer norm
        user_emb = self.layernorm(user_emb)   #128 #user_emb: Tensor(6157,128)
        item_emb = self.layernorm(item_emb)        #item_emb: Tensor(2646,128)

        # obtain users/items embeddings and their interactions
        user_feat = user_emb[user_idx]          #Tensor(128,128)
        # print(user_feat)
        item_feat = item_emb[item_idx]          #Tensor(128,128)
        # print(item_feat)
        interaction = user_feat * item_feat         #Tensor(128,128)  #这里是进行链接预测 如果要进行节点的分类，则直接返回user_emb就好了,如果是做边的回归预测，也是这个，

        # score the node pairs
        pred = self.pred(interaction)        #Tensor(128,128)
        pred = self.dropout(pred)  # dropout
        pred = self.fc(pred)            #Tensor(128,1 )
        pred = torch.sigmoid(pred)    #Tensor(128,1)

        return pred.squeeze(1)

        # return user_emb,item_emb

