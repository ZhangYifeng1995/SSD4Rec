import torch
import copy
from torch import nn
from mamba_ssm import Mamba2
from recbole.model.abstract_recommender import SequentialRecommender

class SSD4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SSD4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"] 
        self.num_layers = config["num_layers"]   
        self.dropout_prob = config["dropout_prob"] 
        self.beta = config["beta"]
        self.norm_embedding = config['norm_embedding']
        
        # Hyperparameters for SSD Block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]  
        self.expand = config["expand"]  
        self.headdim = config['headdim']

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)  # 0 -> mask_token

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.BiSSD_layers = nn.ModuleList([
            BiSSDLayer(
                beta = self.beta,
                d_model=self.hidden_size,  
                d_state=self.d_state,      
                d_conv=self.d_conv,        
                expand=self.expand,        
                dropout=self.dropout_prob, 
                num_layers=self.num_layers,
                headdim = self.headdim
            ) for _ in range(self.num_layers)
        ])

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, cum_item_length, item_idx, flip_index):
        item_emb = self.item_embedding(item_seq)
        if self.norm_embedding == True:
            item_emb = self.dropout(item_emb)
            item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.BiSSD_layers[i](item_emb, item_idx, flip_index)

        # gather_last_token_output
        gather_index = cum_item_length - 1 # [B]
        seq_output = item_emb[0, gather_index, :]

        return seq_output
    
    def calculate_loss(self, item_id, item_id_list, cum_item_length, item_idx, flip_index):
        item_seq = item_id_list.unsqueeze(0)     # [1, cat_dim such as 13297]
        item_idx = item_idx.unsqueeze(0)

        seq_output = self.forward(item_seq, cum_item_length, item_idx, flip_index) # [B, hidden_size]
        pos_items = item_id                      # [B]

        test_item_emb = self.item_embedding.weight # [item_num, hidden_size]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) # [B, item_num]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, item_id_list, cum_item_length, item_idx, flip_index):
        item_seq = item_id_list.unsqueeze(0)    # [1, cat_dim such as 13297]
        item_idx = item_idx.unsqueeze(0)
        
        seq_output = self.forward(item_seq, cum_item_length, item_idx, flip_index) # [B, hidden_size]
        test_items_emb = self.item_embedding.weight # [item_num, hidden_size]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
    
class BiSSDLayer(nn.Module):
    def __init__(self, beta, d_model, d_state, d_conv, expand, dropout, num_layers, headdim):
        super().__init__()
        self.beta = beta

        self.num_layers = num_layers
        self.forward_ssd = Mamba2(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model,
                d_state=d_state,
                headdim = headdim,
                d_conv=d_conv,
                expand=expand,
            )

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    
    def forward(self, item_emb, item_idx, flip_index):
        # forward ssd
        forward_hidden_state = self.forward_ssd(item_emb, seq_idx=item_idx)

        # backward ssd
        filp_emb = item_emb[:, flip_index, :]
        backward_hidden_state = self.forward_ssd(filp_emb, seq_idx=item_idx)

        hidden_states = forward_hidden_state + backward_hidden_state * self.beta + item_emb
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.ffn(hidden_states)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        # Feed-Forward Network
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)

        # residual connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.dropout(hidden_states)

        return hidden_states