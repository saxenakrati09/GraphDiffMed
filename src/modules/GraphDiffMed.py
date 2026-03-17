import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv

from .hetero_effect_graph import hetero_effect_graph
from .homo_relation_graph import homo_relation_graph
from icecream import ic
from sklearn.preprocessing import StandardScaler
import numpy as np

from transformers import DistilBertTokenizer, DistilBertModel

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import multiprocessing as mp

class CausaltyReview(nn.Module):
    def __init__(self, casual_graph, num_diag, num_proc, num_med):
        super(CausaltyReview, self).__init__()

        self.num_med = num_med
        self.c1 = casual_graph
        diag_med_high = casual_graph.get_threshold_effect(0.97, "Diag", "Med")
        diag_med_low = casual_graph.get_threshold_effect(0.90, "Diag", "Med")
        proc_med_high = casual_graph.get_threshold_effect(0.97, "Proc", "Med")
        proc_med_low = casual_graph.get_threshold_effect(0.90, "Proc", "Med")
        self.c1_high_limit = nn.Parameter(torch.tensor([diag_med_high, proc_med_high]))  
        self.c1_low_limit = nn.Parameter(torch.tensor([diag_med_low, proc_med_low]))  
        self.c1_minus_weight = nn.Parameter(torch.tensor(0.01))
        self.c1_plus_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, pre_prob, diags, procs):
        reviewed_prob = pre_prob.clone()

        for m in range(self.num_med):
            max_cdm = 0.0
            max_cpm = 0.0
            for d in diags:
                cdm = self.c1.get_effect(d, m, "Diag", "Med")
                max_cdm = max(max_cdm, cdm)
            for p in procs:
                cpm = self.c1.get_effect(p, m, "Proc", "Med")
                max_cpm = max(max_cpm, cpm)

            if max_cdm < self.c1_low_limit[0] and max_cpm < self.c1_low_limit[1]:
                reviewed_prob[0, m] -= self.c1_minus_weight
            elif max_cdm > self.c1_high_limit[0] or max_cpm > self.c1_high_limit[1]:
                reviewed_prob[0, m] += self.c1_plus_weight

        return reviewed_prob


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GIN, self).__init__()
        nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINConv(nn)

    def forward(self, x, edge_index, weights):
        x = self.conv(x, edge_index)
        weights = weights.unsqueeze(dim=-1)
        x = x * weights
        x_sum = x.sum(dim=0)
        x_sum = x_sum.unsqueeze(dim=0).unsqueeze(dim=0)

        return x_sum

def encode_text(text, device, model_name='distilbert-base-uncased', max_length=512):
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    model.to(device)
    # Split the text into manageable chunks
    tokens = tokenizer(text, add_special_tokens=False)
    input_ids = tokens['input_ids']
    
    chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
    last_hidden_states = []

    
    for chunk in chunks:
        padded_chunk = chunk + [tokenizer.pad_token_id] * (max_length - len(chunk))
        input_ids_chunk = torch.tensor([padded_chunk]).to(device)
        attention_mask = torch.tensor([[1] * len(chunk) + [0] * (max_length - len(chunk))]).to(device)
        
        with torch.no_grad():
            output = model(input_ids_chunk, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state
            last_hidden_states.append(last_hidden_state.squeeze(0))
    del input_ids
    # Concatenate the last hidden states of all chunks
    # Check if last_hidden_states is not empty before concatenating
    if last_hidden_states:
        full_last_hidden_state = torch.cat(last_hidden_states, dim=0)  # [seq_len, 768]
    else:
        # Handle the case when last_hidden_states is empty
        full_last_hidden_state = torch.zeros((1, 768))  # Placeholder tensor with appropriate shape
    del last_hidden_states
    return full_last_hidden_state

def encode_text_(text, device, model_name='distilbert-base-uncased', max_length=512):
    """
    Encode a given text into its last hidden state using a pre-trained DistilBERT model.
    
    Args:
    - text (str): The input text to be encoded.
    - device (torch.device): The device to run the model on.
    - model_name (str): The name of the pre-trained DistilBERT model to use.
    - max_length (int): The maximum length of each chunk of text to be processed.
    
    Returns:
    - torch.tensor: A tensor of shape (seq_len, 768) containing the last hidden state of the input text.
    """
    # Load the pre-trained DistilBERT model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Split the text into manageable chunks
    tokens = tokenizer(text, add_special_tokens=False)
    input_ids = tokens['input_ids']
    
    # Create a batch of all chunks
    chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
    # Clear unnecessary parts of input_ids to save memory
    del input_ids
    input_ids_batch = torch.tensor([chunk + [tokenizer.pad_token_id] * (max_length - len(chunk)) for chunk in chunks]).to(device)
    attention_mask_batch = torch.tensor([[1] * len(chunk) + [0] * (max_length - len(chunk)) for chunk in chunks]).to(device)
    
    # Run the model on the batch of chunks
    with torch.no_grad():
        output = model(input_ids_batch, attention_mask=attention_mask_batch)
        last_hidden_states = output.last_hidden_state
    
    # Concatenate the last hidden states of all chunks
    full_last_hidden_state = torch.cat([last_hidden_state[:len(chunk)] for last_hidden_state, chunk in zip(last_hidden_states, chunks)], dim=0)
    
    # Clear tensors that are no longer needed
    del input_ids_batch, attention_mask_batch, last_hidden_states
    
    return full_last_hidden_state

# class CrossAttention(nn.Module):
#     def __init__(self, emb_dim, num_heads, device, dropout=0.3, batch_first=True):
#         super(CrossAttention, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(emb_dim, num_heads, device=device, dropout=0.3, batch_first=True)

#     def forward(self, query, key, value):
#         # Query: [batch_size, seq_len_query, emb_dim]
#         # Key: [batch_size, seq_len_key, emb_dim]
#         # Value: [batch_size, seq_len_value, emb_dim]
        
#         attn_output, attn_weights = self.multihead_attn(query, key, value)
#         return attn_output



class DiffCrossAttention_intravisit(nn.Module):
    def __init__(self, emb_dim, num_heads, device, dropout=0.3, batch_first=True, lambda_graph=0.1):
        super(DiffCrossAttention_intravisit, self).__init__()
        assert num_heads % 2 == 0, "num_heads must be even for Diff-Attn V2 pairing"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.device = device
        self.dropout = dropout
        self.batch_first = batch_first
        self.lambda_graph = lambda_graph

        # Standard Q/K/V projections - query has 2x heads for differential attention
        self.query_proj = nn.Linear(emb_dim, emb_dim * 2)  # 2x heads for differential pairs
        self.key_proj = nn.Linear(emb_dim, emb_dim)
        self.value_proj = nn.Linear(emb_dim, emb_dim)
        
        # Lambda projection for per-token, per-head gating
        self.lambda_proj = nn.Linear(emb_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def _shape(self, x, B, T, H):
        # (B, T, H*D) -> (B, H, T, D)
        return x.view(B, T, H, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, graph_bias=None):
        B, Tq, _ = query.size()
        _, Tk, _ = key.size()
        
        # Project inputs
        q = self.query_proj(query)  # (B, Tq, 2*emb_dim)
        k = self.key_proj(key)      # (B, Tk, emb_dim) 
        v = self.value_proj(value)  # (B, Tk, emb_dim)
        
        # Reshape for multi-head attention
        q = self._shape(q, B, Tq, self.num_heads * 2)  # (B, 2H, Tq, D)
        k = self._shape(k, B, Tk, self.num_heads)       # (B, H, Tk, D)
        v = self._shape(v, B, Tk, self.num_heads)       # (B, H, Tk, D)
        
        # Repeat k,v for each query head pair
        k = k.repeat_interleave(2, dim=1)  # (B, 2H, Tk, D)
        v = v.repeat_interleave(2, dim=1)  # (B, 2H, Tk, D)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, 2H, Tq, Tk)
        
        # Add graph bias (supports scalar, [Tq,Tk], [B,Tq,Tk], [B,H,Tq,Tk])
        if graph_bias is not None:
            if not torch.is_tensor(graph_bias):
                graph_bias = torch.tensor(graph_bias, device=attn_scores.device, dtype=attn_scores.dtype)
            else:
                graph_bias = graph_bias.to(device=attn_scores.device, dtype=attn_scores.dtype)

            if graph_bias.dim() == 0:
                graph_bias = graph_bias.view(1, 1, 1, 1)
            elif graph_bias.dim() == 2:
                graph_bias = graph_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, Tq, Tk]
            elif graph_bias.dim() == 3:
                graph_bias = graph_bias.unsqueeze(1)  # [B, 1, Tq, Tk]
            elif graph_bias.dim() != 4:
                raise ValueError(f"graph_bias must have 0, 2, 3, or 4 dims, got {graph_bias.dim()}")

            if graph_bias.size(-2) not in (1, Tq) or graph_bias.size(-1) not in (1, Tk):
                raise ValueError(
                    f"graph_bias last dims must be broadcastable to ({Tq}, {Tk}), got ({graph_bias.size(-2)}, {graph_bias.size(-1)})"
                )

            attn_scores = attn_scores + self.lambda_graph * graph_bias
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 2H, Tq, Tk)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # (B, 2H, Tq, D)
        
        # Split into differential pairs: even=branch1, odd=branch2  
        attn1 = context[:, 0::2, :, :]  # (B, H, Tq, D)
        attn2 = context[:, 1::2, :, :]  # (B, H, Tq, D)
        
        # Per-token, per-head lambda: (B, Tq, H) -> (B, H, Tq, 1)
        lambda_val = self.lambda_proj(query)                    # (B, Tq, H)
        lambda_val = torch.sigmoid(lambda_val)                  # gate in (0,1)
        lambda_val = lambda_val.permute(0, 2, 1).unsqueeze(-1) # (B, H, Tq, 1)
        
        # Differential attention: attn1 - λ * attn2
        diff_context = attn1 - lambda_val * attn2              # (B, H, Tq, D)
        
        # Merge heads back
        diff_context = diff_context.transpose(1, 2).contiguous().view(B, Tq, self.emb_dim)  # (B, Tq, emb_dim)
        
        # Output projection with dropout
        output = self.out_proj(self.dropout_layer(diff_context))
        
        return output


class DiffCrossAttention_intervisit(nn.Module):
    def __init__(self, emb_dim, num_heads, device, dropout=0.3, batch_first=True, lambda_graph=0.1):
        super(DiffCrossAttention_intervisit, self).__init__()
        assert num_heads % 2 == 0, "num_heads must be even for Diff-Attn V2 pairing"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.device = device
        self.dropout = dropout
        self.batch_first = batch_first
        self.lambda_graph = lambda_graph

        # Standard Q/K/V projections - query has 2x heads for differential attention
        self.query_proj = nn.Linear(emb_dim, emb_dim * 2)  # 2x heads for differential pairs
        self.key_proj = nn.Linear(emb_dim, emb_dim)
        self.value_proj = nn.Linear(emb_dim, emb_dim)
        
        # Lambda projection for per-token, per-head gating
        self.lambda_proj = nn.Linear(emb_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def _shape(self, x, B, T, H):
        # (B, T, H*D) -> (B, H, T, D)
        return x.view(B, T, H, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, graph_bias=None):
        B, Tq, _ = query.size()
        _, Tk, _ = key.size()
        
        # Project inputs
        q = self.query_proj(query)  # (B, Tq, 2*emb_dim)
        k = self.key_proj(key)      # (B, Tk, emb_dim) 
        v = self.value_proj(value)  # (B, Tk, emb_dim)
        
        # Reshape for multi-head attention
        q = self._shape(q, B, Tq, self.num_heads * 2)  # (B, 2H, Tq, D)
        k = self._shape(k, B, Tk, self.num_heads)       # (B, H, Tk, D)
        v = self._shape(v, B, Tk, self.num_heads)       # (B, H, Tk, D)
        
        # Repeat k,v for each query head pair
        k = k.repeat_interleave(2, dim=1)  # (B, 2H, Tk, D)
        v = v.repeat_interleave(2, dim=1)  # (B, 2H, Tk, D)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, 2H, Tq, Tk)
        
        # Add graph bias (supports scalar, [Tq,Tk], [B,Tq,Tk], [B,H,Tq,Tk])
        if graph_bias is not None:
            if not torch.is_tensor(graph_bias):
                graph_bias = torch.tensor(graph_bias, device=attn_scores.device, dtype=attn_scores.dtype)
            else:
                graph_bias = graph_bias.to(device=attn_scores.device, dtype=attn_scores.dtype)

            if graph_bias.dim() == 0:
                graph_bias = graph_bias.view(1, 1, 1, 1)
            elif graph_bias.dim() == 2:
                graph_bias = graph_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, Tq, Tk]
            elif graph_bias.dim() == 3:
                graph_bias = graph_bias.unsqueeze(1)  # [B, 1, Tq, Tk]
            elif graph_bias.dim() != 4:
                raise ValueError(f"graph_bias must have 0, 2, 3, or 4 dims, got {graph_bias.dim()}")

            if graph_bias.size(-2) not in (1, Tq) or graph_bias.size(-1) not in (1, Tk):
                raise ValueError(
                    f"graph_bias last dims must be broadcastable to ({Tq}, {Tk}), got ({graph_bias.size(-2)}, {graph_bias.size(-1)})"
                )

            attn_scores = attn_scores + self.lambda_graph * graph_bias
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 2H, Tq, Tk)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # (B, 2H, Tq, D)
        
        # Split into differential pairs: even=branch1, odd=branch2  
        attn1 = context[:, 0::2, :, :]  # (B, H, Tq, D)
        attn2 = context[:, 1::2, :, :]  # (B, H, Tq, D)
        
        # Per-token, per-head lambda: (B, Tq, H) -> (B, H, Tq, 1)
        lambda_val = self.lambda_proj(query)                    # (B, Tq, H)
        lambda_val = torch.sigmoid(lambda_val)                  # gate in (0,1)
        lambda_val = lambda_val.permute(0, 2, 1).unsqueeze(-1) # (B, H, Tq, 1)
        
        # Differential attention: attn1 - λ * attn2
        diff_context = attn1 - lambda_val * attn2              # (B, H, Tq, D)
        
        # Merge heads back
        diff_context = diff_context.transpose(1, 2).contiguous().view(B, Tq, self.emb_dim)  # (B, Tq, emb_dim)
        
        # Output projection with dropout
        output = self.out_proj(self.dropout_layer(diff_context))
        
        return output

# class CrossAttention_intervisit(nn.Module):
#     def __init__(self, emb_dim, num_heads, device, dropout=0.3, batch_first=True):
#         super(CrossAttention_intervisit, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(emb_dim, num_heads, device=device, dropout=0.3, batch_first=True)

#     def forward(self, query, key, value):
#         # Query: [batch_size, seq_len_query, emb_dim]
#         # Key: [batch_size, seq_len_key, emb_dim]
#         # Value: [batch_size, seq_len_value, emb_dim]
        
#         attn_output, attn_weights = self.multihead_attn(query, key, value)
#         return attn_output
    
class SelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, device, dropout=0.3, batch_first=True):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(emb_dim, num_heads, device=device, dropout=0.3, batch_first=True)

    def forward(self, query, key, value):
        # Query, Key, Value: [batch_size, seq_len, emb_dim]
        
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        return attn_output
    
class GraphDiffMed(torch.nn.Module):
    """
    GraphDiffMed with Graph-Biased Differential Attention.
    
    This model implements differential attention (v2) with graph-biased structural priors.
    The graph bias from DDI/causal relationships is injected before softmax to preserve
    clinically valid correlations while canceling noise.
    
    Args:
        lambda_graph: Weight for the graph bias term (default: 0.1)
    """
    def __init__(
            self,
            causal_graph,
            mole_relevance,
            tensor_ddi_adj,
            emb_dim,
            voc_size,
            dropout,
            device=torch.device('cpu'),
            lambda_graph=0.1,
    ):
        super(GraphDiffMed, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.lambda_graph = lambda_graph

        # Embedding of all entities
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim),  
            torch.nn.Embedding(voc_size[3], emb_dim),
            torch.nn.Embedding(2, emb_dim),  # Gender (binary, one-hot encoded)
            torch.nn.Linear(1, emb_dim)   # Age (single value)
        ])
                
        # Level 1: Process events within a visit
        self.event_encoder = nn.Sequential(
            nn.Linear(2, emb_dim),  # Input: [ID_normalized, value_normalized]
            nn.ReLU(),
        )      

        # Level 2: Process visits over time
        self.visit_rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=emb_dim,
            batch_first=True,
        )

        self.notes_rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=emb_dim,
            batch_first=True,
        )
        self.notes_encoder = nn.Sequential(
            nn.Linear(768, emb_dim),  # Input: [ID_normalized, value_normalized]
            nn.ReLU(),
        )     
        
        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()

        self.causal_graph = causal_graph

        self.mole_relevance = mole_relevance

        self.mole_med_relevance = torch.tensor(mole_relevance[2])
        self.mole_med_weights = nn.Parameter(torch.ones(mole_relevance[2].shape[1]))

        self.gin_model = GIN(emb_dim, emb_dim)
        # self.crossattn = CrossAttention(emb_dim, 8, device)
        self.selfattn = SelfAttention(emb_dim, 8, device)
        self.diffattn_intervisit = DiffCrossAttention_intervisit(emb_dim, 8, device, lambda_graph=lambda_graph)
        self.diffattn_intravisit = DiffCrossAttention_intravisit(emb_dim, 8, device, lambda_graph=lambda_graph)
        
        self.homo_graph = nn.ModuleList([
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device)
        ])

        self.hetero_graph = torch.nn.ModuleList([
            hetero_effect_graph(emb_dim, emb_dim, device),
            hetero_effect_graph(emb_dim, emb_dim, device),
            hetero_effect_graph(emb_dim, emb_dim, device)
        ])

        # Isomeric and isomeric addition parameters
        self.rho = nn.Parameter(torch.ones(3, 2))

        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.seq_encoders_attn = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        # Convert patient information to drug score
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 8, voc_size[2])
        )
        
        self.query_labevents_notes_demographics = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 14, voc_size[2])
        )
        self.query_notes_demographics = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 12, voc_size[2])
        )
        self.query_labevents_demographics = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 12, voc_size[2])
        )
        self.query_demographics = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 10, voc_size[2])
        )
        self.query_labevents = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 10, voc_size[2])
        )
        self.query_notes = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 10, voc_size[2])
        )
        self.query_labevents_notes = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 12, voc_size[2])
        )
        self.emb_dim = emb_dim

        

        self.review = CausaltyReview(self.causal_graph, voc_size[0], voc_size[1], voc_size[2])

        self.tensor_ddi_adj = tensor_ddi_adj
        # Register DDI adjacency as buffer for graph-biased attention
        self.register_buffer('ddi_graph_bias', torch.FloatTensor(tensor_ddi_adj.cpu().numpy() if isinstance(tensor_ddi_adj, torch.Tensor) else tensor_ddi_adj))
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def _get_batch_graph_bias(self, med_indices_query, med_indices_key=None):
        """
        Extract the relevant DDI subgraph for the given medication indices.
        
        Args:
            med_indices_query: List of medication indices for query (can be from current visit)
            med_indices_key: List of medication indices for key/value (if None, uses same as query)
        
        Returns:
            graph_bias: Tensor of shape [len(med_indices_query), len(med_indices_key)]
                       containing the DDI adjacency values for the given medications
        """
        if med_indices_key is None:
            med_indices_key = med_indices_query
        
        # Convert to tensors if they're lists
        if isinstance(med_indices_query, list):
            med_indices_query = torch.LongTensor(med_indices_query).to(self.device)
        if isinstance(med_indices_key, list):
            med_indices_key = torch.LongTensor(med_indices_key).to(self.device)
        
        # Extract the subgraph from the DDI adjacency matrix
        # Use advanced indexing to get the submatrix
        graph_bias = self.ddi_graph_bias[med_indices_query, :][:, med_indices_key]
        
        return graph_bias

    def _mean_ddi_graph_bias(self, med_indices_query, med_indices_key=None):
        """
        Build a scalar graph bias from DDI adjacency by averaging the relevant subgraph.
        Returns a [1, 1] tensor for easy broadcasting in differential attention.
        """
        if med_indices_query is None or len(med_indices_query) == 0:
            return torch.zeros(1, 1, device=self.device)

        if med_indices_key is None:
            med_indices_key = med_indices_query
        if med_indices_key is None or len(med_indices_key) == 0:
            return torch.zeros(1, 1, device=self.device)

        graph_bias = self._get_batch_graph_bias(med_indices_query, med_indices_key)
        if graph_bias.numel() == 0:
            return torch.zeros(1, 1, device=self.device)

        return graph_bias.float().mean().view(1, 1).to(self.device)

    def _build_intervisit_graph_bias_matrix(
        self,
        query_len,
        key_len,
        query_med_indices,
        key_med_indices_per_visit,
        num_prev_visits,
        query_med_pos=2,
        key_med_block_index=2,
    ):
        """
        Build token-level inter-visit graph bias where only med-to-med token pairs are non-zero.

        Query token order is [diag, proc, med, ...], so default med token index is 2.
        Key token order is concatenated modality blocks over previous visits, with med block index 2.
        """
        graph_bias = torch.zeros(query_len, key_len, device=self.device)

        if num_prev_visits <= 0 or query_med_pos >= query_len:
            return graph_bias

        med_block_start = key_med_block_index * num_prev_visits
        for visit_idx in range(num_prev_visits):
            key_col = med_block_start + visit_idx
            if key_col >= key_len:
                break

            key_visit_meds = key_med_indices_per_visit[visit_idx] if visit_idx < len(key_med_indices_per_visit) else []
            bias_val = self._mean_ddi_graph_bias(query_med_indices, key_visit_meds).squeeze()
            graph_bias[query_med_pos, key_col] = bias_val

        return graph_bias

    def create_graph_data(self, molecule_embeddings):
        num_molecules = len(molecule_embeddings)

        source = []
        target = []
        for i in range(num_molecules):
            for j in range(num_molecules):
                if i != j:
                    source.append(i)
                    target.append(j)

        edge_index = torch.tensor([source, target], dtype=torch.long)

        data = Data(x=molecule_embeddings, edge_index=edge_index)

        return data

    def med_embedding(self, idx_list, emb_mole):
        emb_mole = emb_mole.squeeze(0)
        all_drug_embeddings = []
        for idx in idx_list:
            relevance = self.mole_med_relevance[idx, :].to(self.device)
            mask = relevance != 0
            relevance_masked = relevance.masked_fill(~mask, -float('inf'))
            relevance_normalized = F.softmax(relevance_masked, dim=0)

            relevant_molecule_indices = torch.nonzero(relevance_normalized, as_tuple=True)[0]
            relevant_molecule_embeddings = emb_mole[relevant_molecule_indices]
            weights = self.mole_med_weights[relevant_molecule_indices]
            weights_normalized = F.softmax(weights, dim=0)

            graph_data = self.create_graph_data(relevant_molecule_embeddings)

            drug_embedding = self.gin_model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device), weights_normalized)
            all_drug_embeddings.append(drug_embedding)

        all_drug_embeddings = torch.cat(all_drug_embeddings, dim=1)
        return all_drug_embeddings

    def forward(self, patient_data, args):
        seq_diag, seq_proc, seq_med, seq_labevents, seq_notes = [], [], [], [], []
        seq_med_indices = []  # Store medication indices for each visit

        for adm_id, adm in enumerate(patient_data):
            num_moles = self.embeddings[3].num_embeddings
            idx_mole = torch.arange(num_moles).to(self.device)
            emb_mole = self.embeddings[3](idx_mole).unsqueeze(0)

            idx_diag = torch.LongTensor(adm[0]).to(self.device)
            idx_proc = torch.LongTensor(adm[1]).to(self.device)
            emb_diag = self.rnn_dropout(self.embeddings[0](idx_diag)).unsqueeze(0)
            emb_proc = self.rnn_dropout(self.embeddings[1](idx_proc)).unsqueeze(0)

            relevance_diag = self.mole_relevance[0][adm[0], :]
            emb_diag1 = self.hetero_graph[0](emb_diag, emb_mole, relevance_diag)

            relevance_proc = self.mole_relevance[1][adm[1], :]
            emb_proc1 = self.hetero_graph[1](emb_proc, emb_mole, relevance_proc)

            graph_diag = self.causal_graph.get_graph(adm[6], "Diag")
            graph_proc = self.causal_graph.get_graph(adm[6], "Proc")
            emb_diag2 = self.homo_graph[0](graph_diag, emb_diag1) # [1, seq_lendiag, 64]
            emb_proc2 = self.homo_graph[1](graph_proc, emb_proc1) # [1, seq_lenproc, 64]

            if args.uselabevents:
                # normalizing the input lab events
                # Convert to numpy array
                data = np.array(adm[5])

                # Normalize along axis 1 (row-wise)
                scaler = StandardScaler()
                try:
                    normalized_data = scaler.fit_transform(data)  # Transpose to normalize rows
                    all_visit_embeddings = []
                    for j in range(len(normalized_data)): # labevents
                        # Encode events in visit i
                        event_embeddings = self.event_encoder(torch.Tensor(normalized_data[j]).to(self.device)).unsqueeze(0) # (batch_size, n_events, hidden_dim)
                        all_visit_embeddings.append(event_embeddings)

                    # Stack visit embeddings into a sequence
                    visit_sequence = torch.stack(all_visit_embeddings, dim=1)  # [1, seq_lenlabeevents, 64]
                except:
                    visit_sequence = torch.zeros((1, 1, self.emb_dim)).to(self.device)  

                seq_labevents.append(torch.sum(visit_sequence, keepdim=True, dim=1))

            if args.usenotes:
                # Encoding doctor's notes using BERT
                notes = ' '.join(adm[4])
                # remove all whitespace characters
                notes = ' '.join(notes.split())
                result = encode_text(notes, self.device) # [seq_len, 768]
                result = result.unsqueeze(0) # [1, seq_len, 768]
                result = self.notes_encoder(result.to(self.device))

                # This is why we get memory error
                # seq_notes.append(torch.sum(result, keepdim=True, dim=1))
                # The above line is storing the entire sequence in memory
                # Instead, we should store only the sum of the sequence
                seq_notes.append(torch.sum(result, dim=1).unsqueeze(0))
                del result
            if adm == patient_data[0]:
                emb_med2 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient_data[adm_id - 1]
                emb_med1 = self.rnn_dropout(self.med_embedding(adm_last[2], emb_mole))

                med_graph = self.causal_graph.get_graph(adm_last[6], "Med")
                emb_med2 = self.homo_graph[2](med_graph, emb_med1)

            seq_diag.append(torch.sum(emb_diag2, keepdim=True, dim=1))
            seq_proc.append(torch.sum(emb_proc2, keepdim=True, dim=1))
            seq_med.append(torch.sum(emb_med2, keepdim=True, dim=1))
            # Store current visit's medication indices
            if adm == patient_data[0]:
                seq_med_indices.append([])  # First visit has no previous meds
            else:
                seq_med_indices.append(patient_data[adm_id - 1][2])  # Previous visit's meds

        seq_diag = torch.cat(seq_diag, dim=1) #[1, visits, 64]
        seq_proc = torch.cat(seq_proc, dim=1)
        seq_med = torch.cat(seq_med, dim=1)

        # Graph bias derived from medication sets per visit
        intravisit_graph_biases = [self._mean_ddi_graph_bias(visit_meds) for visit_meds in seq_med_indices]

        
        output_diag, hidden_diag = self.seq_encoders[0](seq_diag)
        output_proc, hidden_proc = self.seq_encoders[1](seq_proc)
        output_med, hidden_med = self.seq_encoders[2](seq_med)
            
        if args.usedemographics and not args.uselabevents and not args.usenotes: # only demographics
            # Embeddings for patient information
            gender_idx = torch.LongTensor([1 if adm[3][0] == 'F' else 0]).to(self.device)
            gender_emb = self.rnn_dropout(self.embeddings[4](gender_idx)).unsqueeze(0)

            age_tensor = torch.FloatTensor([adm[3][1]]).unsqueeze(0).to(self.device)
            age_emb = self.rnn_dropout(self.embeddings[5](age_tensor)).unsqueeze(0)
            
            if seq_diag.shape[1] > 1:
                intravisit_attn_scores_list = []
                for i in range(seq_med.shape[1]):
                    kv_diag = seq_diag[:, i, :].unsqueeze(1)
                    kv_proc = seq_proc[:, i, :].unsqueeze(1)
                    intravisit_attn_scores_diag = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_diag, kv_diag, intravisit_graph_biases[i])
                    intravisit_attn_scores_proc = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_proc, kv_proc, intravisit_graph_biases[i])
                    intravisit_attn_scores = torch.cat((intravisit_attn_scores_diag, intravisit_attn_scores_proc), dim=1)
                    intravisit_attn_scores_list.append(intravisit_attn_scores)
                intravisit_attn_scores = torch.cat(intravisit_attn_scores_list, dim=1)
            else:
                intravisit_attn_scores = torch.zeros_like(seq_med)
            intravisit_attn_scores = torch.sum(intravisit_attn_scores, keepdim=True, dim=1)
            
            if output_diag.shape[1] > 1: # if only one visit
                previous_output_diag = output_diag[:, :-1, :]
                previous_output_proc = output_proc[:, :-1, :]
                previous_output_med = output_med[:, :-1, :]
                
                # TODO: add inter-visit attention here
                visit_emb_last = torch.cat([seq_diag[:, -1:, :], seq_proc[:, -1:, :], seq_med[:, -1:, :], gender_emb, age_emb], dim=1)
                previous_gru_output = torch.cat([previous_output_diag, previous_output_proc, previous_output_med], dim=1) 
                intervisit_graph_bias_matrix = self._build_intervisit_graph_bias_matrix(
                    query_len=visit_emb_last.size(1),
                    key_len=previous_gru_output.size(1),
                    query_med_indices=seq_med_indices[-1] if len(seq_med_indices) > 0 else [],
                    key_med_indices_per_visit=seq_med_indices[:-1],
                    num_prev_visits=previous_output_diag.shape[1],
                )
                interattn_scores = self.diffattn_intervisit(visit_emb_last, previous_gru_output, previous_gru_output, intervisit_graph_bias_matrix)
                interattn_scores = torch.sum(interattn_scores, keepdim=True, dim=1)

            else:
                interattn_scores = torch.zeros(1, 1, 64).to(self.device)
            seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med, gender_emb, age_emb, interattn_scores, intravisit_attn_scores], dim=-1)
            last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1]], dim=-1)
            patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
        
            score = self.query_demographics(patient_repr).unsqueeze(0)
        elif args.uselabevents and not args.usenotes and not args.usedemographics: # only labevents

            seq_labevents = torch.cat(seq_labevents, dim=1)
            output_labevents, hidden_labevents = self.visit_rnn(seq_labevents)

            if seq_diag.shape[1] > 1:
                intravisit_attn_scores_list = []
                for i in range(seq_med.shape[1]):
                    kv_diag = seq_diag[:, i, :].unsqueeze(1)
                    kv_proc = seq_proc[:, i, :].unsqueeze(1)
                    kv_labevents = seq_labevents[:, i, :].unsqueeze(1)
                    intravisit_attn_scores_diag = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_diag, kv_diag, intravisit_graph_biases[i])
                    intravisit_attn_scores_proc = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_proc, kv_proc, intravisit_graph_biases[i])
                    intravisit_attn_scores_labevents = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_labevents, kv_labevents, intravisit_graph_biases[i])
                    intravisit_attn_scores = torch.cat((intravisit_attn_scores_diag, intravisit_attn_scores_proc, intravisit_attn_scores_labevents), dim=1)
                    intravisit_attn_scores_list.append(intravisit_attn_scores)
                intravisit_attn_scores = torch.cat(intravisit_attn_scores_list, dim=1)
            else:
                intravisit_attn_scores = torch.zeros_like(seq_med)
            intravisit_attn_scores = torch.sum(intravisit_attn_scores, keepdim=True, dim=1)
            
            if output_diag.shape[1] > 1: # if only one visit
                previous_output_diag = output_diag[:, :-1, :]
                previous_output_proc = output_proc[:, :-1, :]
                previous_output_med = output_med[:, :-1, :]
                previous_output_labevents = output_labevents[:, :-1, :]
                
                # TODO: add inter-visit attention here
                visit_emb_last = torch.cat([seq_diag[:, -1:, :], seq_proc[:, -1:, :], seq_med[:, -1:, :], seq_labevents[:, -1:, :]], dim=1)
                previous_gru_output = torch.cat([previous_output_diag, previous_output_proc, previous_output_med, previous_output_labevents], dim=1) 
                intervisit_graph_bias_matrix = self._build_intervisit_graph_bias_matrix(
                    query_len=visit_emb_last.size(1),
                    key_len=previous_gru_output.size(1),
                    query_med_indices=seq_med_indices[-1] if len(seq_med_indices) > 0 else [],
                    key_med_indices_per_visit=seq_med_indices[:-1],
                    num_prev_visits=previous_output_diag.shape[1],
                )
                interattn_scores = self.diffattn_intervisit(visit_emb_last, previous_gru_output, previous_gru_output, intervisit_graph_bias_matrix)
                interattn_scores = torch.sum(interattn_scores, keepdim=True, dim=1)
            else:
                interattn_scores = torch.zeros(1, 1, 64).to(self.device)
                
            seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med, hidden_labevents, interattn_scores, intravisit_attn_scores], dim=-1)
            last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1], output_labevents[:, -1]], dim=-1)
            patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
            
            score = self.query_labevents(patient_repr).unsqueeze(0)
        elif args.usenotes and not args.uselabevents and not args.usedemographics: # only notes

            seq_notes = torch.cat(seq_notes, dim=1)
            output_notes, hidden_notes = self.notes_rnn(seq_notes)

            if seq_diag.shape[1] > 1:
                intravisit_attn_scores_list = []
                for i in range(seq_med.shape[1]):
                    kv_diag = seq_diag[:, i, :].unsqueeze(1)
                    kv_proc = seq_proc[:, i, :].unsqueeze(1)
                    kv_notes = seq_notes[:, i, :].unsqueeze(1)
                    intravisit_attn_scores_diag = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_diag, kv_diag, intravisit_graph_biases[i])
                    intravisit_attn_scores_proc = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_proc, kv_proc, intravisit_graph_biases[i])
                    intravisit_attn_scores_notes = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_notes, kv_notes, intravisit_graph_biases[i])
                    intravisit_attn_scores = torch.cat((intravisit_attn_scores_diag, intravisit_attn_scores_proc, intravisit_attn_scores_notes), dim=1)
                    intravisit_attn_scores_list.append(intravisit_attn_scores)
                intravisit_attn_scores = torch.cat(intravisit_attn_scores_list, dim=1)
            else:
                intravisit_attn_scores = torch.zeros_like(seq_med)
            intravisit_attn_scores = torch.sum(intravisit_attn_scores, keepdim=True, dim=1)
            
            if output_diag.shape[1] > 1: # if only one visit
                previous_output_diag = output_diag[:, :-1, :]
                previous_output_proc = output_proc[:, :-1, :]
                previous_output_med = output_med[:, :-1, :]
                previous_output_notes = output_notes[:, :-1, :]
                
                # TODO: add inter-visit attention here
                visit_emb_last = torch.cat([seq_diag[:, -1:, :], seq_proc[:, -1:, :], seq_med[:, -1:, :], seq_notes[:, -1:, :]], dim=1)
                previous_gru_output = torch.cat([previous_output_diag, previous_output_proc, previous_output_med, previous_output_notes], dim=1) 
                intervisit_graph_bias_matrix = self._build_intervisit_graph_bias_matrix(
                    query_len=visit_emb_last.size(1),
                    key_len=previous_gru_output.size(1),
                    query_med_indices=seq_med_indices[-1] if len(seq_med_indices) > 0 else [],
                    key_med_indices_per_visit=seq_med_indices[:-1],
                    num_prev_visits=previous_output_diag.shape[1],
                )
                interattn_scores = self.diffattn_intervisit(visit_emb_last, previous_gru_output, previous_gru_output, intervisit_graph_bias_matrix)
                interattn_scores = torch.sum(interattn_scores, keepdim=True, dim=1)

            else:
                interattn_scores = torch.zeros(1, 1, 64).to(self.device)
            
            seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med, hidden_notes, interattn_scores, intravisit_attn_scores], dim=-1)
            last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1], output_notes[:, -1]], dim=-1)
            patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
            score = self.query_notes(patient_repr).unsqueeze(0)
        elif args.uselabevents and args.usenotes and not args.usedemographics: # only labevents and notes

            seq_notes = torch.cat(seq_notes, dim=1)
            output_notes, hidden_notes = self.notes_rnn(seq_notes)
            
            seq_labevents = torch.cat(seq_labevents, dim=1)
            output_labevents, hidden_labevents = self.visit_rnn(seq_labevents)

            if seq_diag.shape[1] > 1:
                intravisit_attn_scores_list = []
                for i in range(seq_med.shape[1]):
                    kv_diag = seq_diag[:, i, :].unsqueeze(1)
                    kv_proc = seq_proc[:, i, :].unsqueeze(1)
                    kv_labevents = seq_labevents[:, i, :].unsqueeze(1)
                    kv_notes = seq_notes[:, i, :].unsqueeze(1)

                    intravisit_attn_scores_diag = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_diag, kv_diag, intravisit_graph_biases[i])
                    intravisit_attn_scores_proc = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_proc, kv_proc, intravisit_graph_biases[i])
                    intravisit_attn_scores_labevents = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_labevents, kv_labevents, intravisit_graph_biases[i])
                    intravisit_attn_scores_notes = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_notes, kv_notes, intravisit_graph_biases[i])

                    intravisit_attn_scores = torch.cat((
                        intravisit_attn_scores_diag, 
                        intravisit_attn_scores_proc, 
                        intravisit_attn_scores_labevents,
                        intravisit_attn_scores_notes
                    ), dim=1)
                    intravisit_attn_scores_list.append(intravisit_attn_scores)
                intravisit_attn_scores = torch.cat(intravisit_attn_scores_list, dim=1)
            else:
                intravisit_attn_scores = torch.zeros_like(seq_med)
            intravisit_attn_scores = torch.sum(intravisit_attn_scores, keepdim=True, dim=1)
            
            if output_diag.shape[1] > 1: # if only one visit
                previous_output_diag = output_diag[:, :-1, :]
                previous_output_proc = output_proc[:, :-1, :]
                previous_output_med = output_med[:, :-1, :]
                previous_output_labevents = output_labevents[:, :-1, :]
                previous_output_notes = output_notes[:, :-1, :]
                
                # TODO: add inter-visit attention here
                visit_emb_last = torch.cat([seq_diag[:, -1:, :], seq_proc[:, -1:, :], seq_med[:, -1:, :], seq_labevents[:, -1:, :], seq_notes[:, -1:, :]], dim=1)
                previous_gru_output = torch.cat([previous_output_diag, previous_output_proc, previous_output_med, previous_output_labevents, previous_output_notes], dim=1) 
                intervisit_graph_bias_matrix = self._build_intervisit_graph_bias_matrix(
                    query_len=visit_emb_last.size(1),
                    key_len=previous_gru_output.size(1),
                    query_med_indices=seq_med_indices[-1] if len(seq_med_indices) > 0 else [],
                    key_med_indices_per_visit=seq_med_indices[:-1],
                    num_prev_visits=previous_output_diag.shape[1],
                )
                interattn_scores = self.diffattn_intervisit(visit_emb_last, previous_gru_output, previous_gru_output, intervisit_graph_bias_matrix)
                interattn_scores = torch.sum(interattn_scores, keepdim=True, dim=1)

            else:
                interattn_scores = torch.zeros(1, 1, 64).to(self.device)
            
            seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med, hidden_labevents, hidden_notes, interattn_scores, intravisit_attn_scores], dim=-1)
            last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1], output_labevents[:, -1], output_notes[:, -1]], dim=-1)
            patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
            score = self.query_labevents_notes(patient_repr).unsqueeze(0)
        elif args.uselabevents and args.usedemographics and not args.usenotes: # only labevents and demographics

            seq_labevents = torch.cat(seq_labevents, dim=1)
            output_labevents, hidden_labevents = self.visit_rnn(seq_labevents)
            # Embeddings for patient information
            gender_idx = torch.LongTensor([1 if adm[3][0] == 'F' else 0]).to(self.device)
            gender_emb = self.rnn_dropout(self.embeddings[4](gender_idx)).unsqueeze(0)

            age_tensor = torch.FloatTensor([adm[3][1]]).unsqueeze(0).to(self.device)
            age_emb = self.rnn_dropout(self.embeddings[5](age_tensor)).unsqueeze(0)

            if seq_diag.shape[1] > 1:
                intravisit_attn_scores_list = []
                for i in range(seq_med.shape[1]):
                    kv_diag = seq_diag[:, i, :].unsqueeze(1)
                    kv_proc = seq_proc[:, i, :].unsqueeze(1)
                    kv_labevents = seq_labevents[:, i, :].unsqueeze(1)
                    intravisit_attn_scores_diag = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_diag, kv_diag, intravisit_graph_biases[i])
                    intravisit_attn_scores_proc = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_proc, kv_proc, intravisit_graph_biases[i])
                    intravisit_attn_scores_labevents = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_labevents, kv_labevents, intravisit_graph_biases[i])
                    intravisit_attn_scores = torch.cat((intravisit_attn_scores_diag, intravisit_attn_scores_proc, intravisit_attn_scores_labevents), dim=1)
                    intravisit_attn_scores_list.append(intravisit_attn_scores)
                intravisit_attn_scores = torch.cat(intravisit_attn_scores_list, dim=1)
            else:
                intravisit_attn_scores = torch.zeros_like(seq_med)
            intravisit_attn_scores = torch.sum(intravisit_attn_scores, keepdim=True, dim=1)
            
            
            if output_diag.shape[1] > 1: # if only one visit
                previous_output_diag = output_diag[:, :-1, :]
                previous_output_proc = output_proc[:, :-1, :]
                previous_output_med = output_med[:, :-1, :]
                previous_output_labevents = output_labevents[:, :-1, :]
                
                # TODO: add inter-visit attention here
                visit_emb_last = torch.cat([seq_diag[:, -1:, :], seq_proc[:, -1:, :], seq_med[:, -1:, :], seq_labevents[:, -1:, :], gender_emb, age_emb], dim=1)
                previous_gru_output = torch.cat([previous_output_diag, previous_output_proc, previous_output_med, previous_output_labevents], dim=1) 
                intervisit_graph_bias_matrix = self._build_intervisit_graph_bias_matrix(
                    query_len=visit_emb_last.size(1),
                    key_len=previous_gru_output.size(1),
                    query_med_indices=seq_med_indices[-1] if len(seq_med_indices) > 0 else [],
                    key_med_indices_per_visit=seq_med_indices[:-1],
                    num_prev_visits=previous_output_diag.shape[1],
                )
                interattn_scores = self.diffattn_intervisit(visit_emb_last, previous_gru_output, previous_gru_output, intervisit_graph_bias_matrix)
                interattn_scores = torch.sum(interattn_scores, keepdim=True, dim=1)

            else:
                interattn_scores = torch.zeros(1, 1, 64).to(self.device)
                
            seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med, hidden_labevents, gender_emb, age_emb, interattn_scores, intravisit_attn_scores], dim=-1)
            last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1], output_labevents[:, -1]], dim=-1)
            patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
            score = self.query_labevents_demographics(patient_repr).unsqueeze(0)
        elif args.usenotes and args.usedemographics and not args.uselabevents: # only notes and demographics

            seq_notes = torch.cat(seq_notes, dim=1)
            output_notes, hidden_notes = self.notes_rnn(seq_notes)
            
            # Embeddings for patient information
            gender_idx = torch.LongTensor([1 if adm[3][0] == 'F' else 0]).to(self.device)
            gender_emb = self.rnn_dropout(self.embeddings[4](gender_idx)).unsqueeze(0)

            age_tensor = torch.FloatTensor([adm[3][1]]).unsqueeze(0).to(self.device)
            age_emb = self.rnn_dropout(self.embeddings[5](age_tensor)).unsqueeze(0)
                
            if seq_diag.shape[1] > 1:
                intravisit_attn_scores_list = []
                for i in range(seq_med.shape[1]):
                    kv_diag = seq_diag[:, i, :].unsqueeze(1)
                    kv_proc = seq_proc[:, i, :].unsqueeze(1)
                    kv_notes = seq_notes[:, i, :].unsqueeze(1)
                    intravisit_attn_scores_diag = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_diag, kv_diag, intravisit_graph_biases[i])
                    intravisit_attn_scores_proc = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_proc, kv_proc, intravisit_graph_biases[i])
                    intravisit_attn_scores_notes = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_notes, kv_notes, intravisit_graph_biases[i])
                    intravisit_attn_scores = torch.cat((intravisit_attn_scores_diag, intravisit_attn_scores_proc, intravisit_attn_scores_notes), dim=1)
                    intravisit_attn_scores_list.append(intravisit_attn_scores)
                intravisit_attn_scores = torch.cat(intravisit_attn_scores_list, dim=1)
            else:
                intravisit_attn_scores = torch.zeros_like(seq_med)
            intravisit_attn_scores = torch.sum(intravisit_attn_scores, keepdim=True, dim=1)
            
            if output_diag.shape[1] > 1: # if only one visit
                previous_output_diag = output_diag[:, :-1, :]
                previous_output_proc = output_proc[:, :-1, :]
                previous_output_med = output_med[:, :-1, :]
                previous_output_notes = output_notes[:, :-1, :]
                
                # TODO: add inter-visit attention here
                visit_emb_last = torch.cat([seq_diag[:, -1:, :], seq_proc[:, -1:, :], seq_med[:, -1:, :], seq_notes[:, -1:, :], gender_emb, age_emb], dim=1)
                previous_gru_output = torch.cat([previous_output_diag, previous_output_proc, previous_output_med, previous_output_notes], dim=1) 
                intervisit_graph_bias_matrix = self._build_intervisit_graph_bias_matrix(
                    query_len=visit_emb_last.size(1),
                    key_len=previous_gru_output.size(1),
                    query_med_indices=seq_med_indices[-1] if len(seq_med_indices) > 0 else [],
                    key_med_indices_per_visit=seq_med_indices[:-1],
                    num_prev_visits=previous_output_diag.shape[1],
                )
                interattn_scores = self.diffattn_intervisit(visit_emb_last, previous_gru_output, previous_gru_output, intervisit_graph_bias_matrix)
                interattn_scores = torch.sum(interattn_scores, keepdim=True, dim=1)

            else:
                interattn_scores = torch.zeros(1, 1, 64).to(self.device)
                
            seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med, hidden_notes, gender_emb, age_emb, interattn_scores, intravisit_attn_scores], dim=-1)
            last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1], output_notes[:, -1]], dim=-1)
            patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
            score = self.query_notes_demographics(patient_repr).unsqueeze(0)
        elif args.uselabevents and args.usenotes and args.usedemographics: # labevents and notes and demographics

            seq_notes = torch.cat(seq_notes, dim=1)
            output_notes, hidden_notes = self.notes_rnn(seq_notes)
            
            seq_labevents = torch.cat(seq_labevents, dim=1)
            output_labevents, hidden_labevents = self.visit_rnn(seq_labevents)
            # Embeddings for patient information
            gender_idx = torch.LongTensor([1 if adm[3][0] == 'F' else 0]).to(self.device)
            gender_emb = self.rnn_dropout(self.embeddings[4](gender_idx)).unsqueeze(0)

            age_tensor = torch.FloatTensor([adm[3][1]]).unsqueeze(0).to(self.device)
            age_emb = self.rnn_dropout(self.embeddings[5](age_tensor)).unsqueeze(0)
            
            if seq_diag.shape[1] > 1: # if more than one visit
                intravisit_attn_scores_list = []
                for i in range(seq_med.shape[1]):
                    kv_diag = seq_diag[:, i, :].unsqueeze(1)
                    kv_proc = seq_proc[:, i, :].unsqueeze(1)
                    kv_labevents = seq_labevents[:, i, :].unsqueeze(1)
                    kv_notes = seq_notes[:, i, :].unsqueeze(1)
                    intravisit_attn_scores_diag = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_diag, kv_diag, intravisit_graph_biases[i])
                    intravisit_attn_scores_proc = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_proc, kv_proc, intravisit_graph_biases[i])
                    intravisit_attn_scores_labevents = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_labevents, kv_labevents, intravisit_graph_biases[i])
                    intravisit_attn_scores_notes = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_notes, kv_notes, intravisit_graph_biases[i])
                    intravisit_attn_scores = torch.cat((intravisit_attn_scores_diag, intravisit_attn_scores_proc, intravisit_attn_scores_labevents, intravisit_attn_scores_notes), dim=1)
                    intravisit_attn_scores_list.append(intravisit_attn_scores)
                intravisit_attn_scores = torch.cat(intravisit_attn_scores_list, dim=1)
            else:
                intravisit_attn_scores = torch.zeros_like(seq_med)
            intravisit_attn_scores = torch.sum(intravisit_attn_scores, keepdim=True, dim=1)
            
            if output_diag.shape[1] > 1: # if only one visit
                previous_output_diag = output_diag[:, :-1, :]
                previous_output_proc = output_proc[:, :-1, :]
                previous_output_med = output_med[:, :-1, :]
                previous_output_labevents = output_labevents[:, :-1, :]
                previous_output_notes = output_notes[:, :-1, :]
                
                # TODO: add inter-visit attention here
                visit_emb_last = torch.cat([seq_diag[:, -1:, :], seq_proc[:, -1:, :], seq_med[:, -1:, :], seq_labevents[:, -1:, :], seq_notes[:, -1:, :], gender_emb, age_emb], dim=1)
                previous_gru_output = torch.cat([previous_output_diag, previous_output_proc, previous_output_med, previous_output_labevents, previous_output_notes], dim=1) 
                intervisit_graph_bias_matrix = self._build_intervisit_graph_bias_matrix(
                    query_len=visit_emb_last.size(1),
                    key_len=previous_gru_output.size(1),
                    query_med_indices=seq_med_indices[-1] if len(seq_med_indices) > 0 else [],
                    key_med_indices_per_visit=seq_med_indices[:-1],
                    num_prev_visits=previous_output_diag.shape[1],
                )
                interattn_scores = self.diffattn_intervisit(visit_emb_last, previous_gru_output, previous_gru_output, intervisit_graph_bias_matrix)
                interattn_scores = torch.sum(interattn_scores, keepdim=True, dim=1)

            else:
                interattn_scores = torch.zeros(1, 1, 64).to(self.device)
            seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med, hidden_labevents, hidden_notes, gender_emb, age_emb, interattn_scores, intravisit_attn_scores], dim=-1)
            last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1], output_labevents[:, -1], output_notes[:, -1]], dim=-1)

            patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
            score = self.query_labevents_notes_demographics(patient_repr).unsqueeze(0)
        else: # none
            if seq_diag.shape[1] > 1:
                intravisit_attn_scores_list = []
                for i in range(seq_med.shape[1]):
                    kv_diag = seq_diag[:, i, :].unsqueeze(1)
                    kv_proc = seq_proc[:, i, :].unsqueeze(1)
                    intravisit_attn_scores_diag = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_diag, kv_diag, intravisit_graph_biases[i])
                    intravisit_attn_scores_proc = self.diffattn_intravisit(seq_med[:, i, :].unsqueeze(1), kv_proc, kv_proc, intravisit_graph_biases[i])
                    intravisit_attn_scores = torch.cat((intravisit_attn_scores_diag, intravisit_attn_scores_proc), dim=1)
                    intravisit_attn_scores_list.append(intravisit_attn_scores)
                intravisit_attn_scores = torch.cat(intravisit_attn_scores_list, dim=1)
            else:
                intravisit_attn_scores = torch.zeros_like(seq_med)
            intravisit_attn_scores = torch.sum(intravisit_attn_scores, keepdim=True, dim=1)
            if output_diag.shape[1] > 1: # if only one visit
                previous_output_diag = output_diag[:, :-1, :]
                previous_output_proc = output_proc[:, :-1, :]
                previous_output_med = output_med[:, :-1, :]
                
                # TODO: add inter-visit attention here
                visit_emb_last = torch.cat([seq_diag[:, -1:, :], seq_proc[:, -1:, :], seq_med[:, -1:, :]], dim=1)
                previous_gru_output = torch.cat([previous_output_diag, previous_output_proc, previous_output_med], dim=1) 
                intervisit_graph_bias_matrix = self._build_intervisit_graph_bias_matrix(
                    query_len=visit_emb_last.size(1),
                    key_len=previous_gru_output.size(1),
                    query_med_indices=seq_med_indices[-1] if len(seq_med_indices) > 0 else [],
                    key_med_indices_per_visit=seq_med_indices[:-1],
                    num_prev_visits=previous_output_diag.shape[1],
                )
                interattn_scores = self.diffattn_intervisit(visit_emb_last, previous_gru_output, previous_gru_output, intervisit_graph_bias_matrix)
                interattn_scores = torch.sum(interattn_scores, keepdim=True, dim=1)

            else:
                interattn_scores = torch.zeros(1, 1, self.emb_dim).to(self.device)
                
            seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med, interattn_scores, intravisit_attn_scores], dim=-1)
            last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1]], dim=-1)

            patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
            score = self.query(patient_repr).unsqueeze(0)


        score = self.review(score, patient_data[-1][0], patient_data[-1][1])

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return score, batch_neg
    
