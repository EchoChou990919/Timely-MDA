import torch
from torch import Tensor
import torch.nn as nn

from torch_geometric.data import HeteroData

from torch_geometric.nn import HGTConv

class Embedding(torch.nn.Module):
    def __init__(self, dim, ablation_type):
        super().__init__()
        
        self.mirna_emb = nn.Linear(640, dim)
        self.disease_emb = nn.Linear(768, dim)
        self.ablation_type = ablation_type
        if ablation_type > 2:
            self.pcg_emb = nn.Linear(768, dim)
    
    def forward(self, x_dict):
        
        mirna_x = self.mirna_emb(x_dict['mirna'])
        disease_x = self.disease_emb(x_dict['disease'])
        
        embedding_x_dict = {
            'mirna': mirna_x,
            'disease': disease_x
        }
        
        if self.ablation_type > 2:
            pcg_x = self.pcg_emb(x_dict['pcg'])
            embedding_x_dict['pcg'] = pcg_x
            
        return embedding_x_dict


# Prepared for ablation study, but not used in the final model
class GNN(torch.nn.Module):
    def __init__(self, dim, num_heads, num_layers, group_type, feature_ablation_type):
        super().__init__()

        self.feature_ablation_type = feature_ablation_type

        metadata = (['mirna', 'disease'],
                    [('mirna', 'family', 'mirna'),
                     ('disease', 'fatherson', 'disease')])
        
        if feature_ablation_type > 2:
            metadata[0].append('pcg')
            metadata[1].append(('pcg', 'interaction', 'pcg'))
            metadata[1].append(('mirna', 'association', 'pcg'))
            metadata[1].append(('pcg', 'association', 'disease'))
            metadata[1].append(('pcg', 'rev_association', 'mirna'))
            metadata[1].append(('disease', 'rev_association', 'pcg'))
        
        if feature_ablation_type > 3:
            metadata[1].append(('mirna', 'association', 'disease'))
            metadata[1].append(('disease', 'rev_association', 'mirna'))
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(in_channels=dim, out_channels=dim, metadata=metadata, heads=num_heads, group=group_type)
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        
        conv_edge_index_dict = {
            ('mirna', 'family', 'mirna'): edge_index_dict[('mirna', 'family', 'mirna')],
            ('disease', 'fatherson', 'disease'): edge_index_dict[('disease', 'fatherson', 'disease')],
        }
        
        if self.feature_ablation_type > 2:
            conv_edge_index_dict[('mirna', 'association', 'pcg')] = edge_index_dict[('mirna', 'association', 'pcg')]
            conv_edge_index_dict[('pcg', 'rev_association', 'mirna')] = edge_index_dict[('pcg', 'rev_association', 'mirna')]
            conv_edge_index_dict[('pcg', 'association', 'disease')] = edge_index_dict[('pcg', 'association', 'disease')]
            conv_edge_index_dict[('disease', 'rev_association', 'pcg')] = edge_index_dict[('disease', 'rev_association', 'pcg')]
            conv_edge_index_dict[('pcg', 'interaction', 'pcg')] = edge_index_dict[('pcg', 'interaction', 'pcg')]
        
        if self.feature_ablation_type > 3:
            conv_edge_index_dict[('mirna', 'association', 'disease')] = edge_index_dict[('mirna', 'association', 'disease')]
            conv_edge_index_dict[('disease', 'rev_association', 'mirna')] = edge_index_dict[('disease', 'rev_association', 'mirna')]
        
        for conv in self.convs:
            x_dict = conv(x_dict, conv_edge_index_dict)
        
        return x_dict


class Classifier(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_mirna, x_disease, edge_label_index):
        
        edge_feature_mirna = x_mirna[edge_label_index[0]]
        edge_feature_disease = x_disease[edge_label_index[1]]
        
        edge_feature = torch.cat([edge_feature_mirna, edge_feature_disease], 1)
        result = self.mlp_layers(edge_feature).squeeze(-1)
        
        return result
    
# Ablation Tpye: 
# 1: miRNA sequence / disease text
# 2: 1 + miRNA family associations / disease fatherson associations
# **3**: 1 + 2 + miRNA / disease - pcg associations --> the only used final model!
# 4: 1 + 2 + 3 + existing miRNA - disease associations

class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, num_layers, group_type, feature_ablation_type):
        super().__init__()
        
        self.feature_ablation_type = feature_ablation_type
        self.embeddings = Embedding(dim, feature_ablation_type)
        if feature_ablation_type > 1:
            self.gnn = GNN(dim, num_heads, num_layers, group_type, feature_ablation_type)
        self.classifier = Classifier(dim)

    def forward(self, data: HeteroData) -> Tensor:
        
        x_dict = {
            'mirna': data['mirna'].x.float(),
            'disease': data['disease'].x.float()
        }
        if self.feature_ablation_type > 2:
            x_dict['pcg'] = data['pcg'].x.float()
        
        x_dict = self.embeddings(x_dict)
        
        if self.feature_ablation_type > 1:
            x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict['mirna'],
            x_dict['disease'],
            data['mirna', 'association', 'disease'].edge_label_index,
        )

        return pred