import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder
from einops import rearrange
# Antnas
# from tabpfn.layer import TransformerEncoderLayer, _get_activation_fn
# from tabpfn.utils import SeqBN, bool_mask_to_att_mask
from layer import TransformerEncoderLayer, _get_activation_fn
from utils import SeqBN, bool_mask_to_att_mask



class TransformerModel(nn.Module):
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers, dropout=0.0, style_encoder=None, y_encoder=None,
                 pos_encoder=None, decoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, num_global_att_tokens=0, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True):
        super().__init__()
        self.model_type = 'Transformer'
        
        encoder_layer_creator = lambda: TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                pre_norm=pre_norm, recompute_attn=recompute_attn)
        
        # Initiate n subsequent layers of transformer (initiated all the same or not)
        # all_layers_same_init=False by default and not changed later so we do TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), 3)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, 3)
        self.ninp = ninp
        
        # Store the encoder, decoder modules
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.decoder = decoder(ninp, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        if num_global_att_tokens is not None: 
            assert not full_attention
        
        self.global_att_embeddings = nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None # seems like global_att_embeddings=None
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking

        self.n_out = n_out
        self.nhid = nhid

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('efficient_eval_masking', False)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generates an upper triangular matrix with -inf and 0.0

        Args:
            sz (int): Batch size

        Returns:
            tensor: mask - upper triangular matrix
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        """Generates same attnetion matrix as in paper (first one
        with the diagonal being one) except all 1 entries are 0.0 
        and 0 entries are -inf

        Args:
            sz (int): batch size
            query_size (int): number of query

        Returns:
            tensor: mask that masks y but attends itself (diagonal 0.0 NOT -inf)
        """
        train_size = sz-query_size
        mask = torch.zeros(sz,sz) == 0
        mask[:,train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_query_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        """Generates matrix with row for each query explaining which points it should attend. Includes itself.

        Args:
            num_global_att_tokens (int): 
            seq_len (int): number of points in batch (I believe)
            num_query_tokens (int): 

        Returns:
            mask: num_query_tokens x (seq_len + num_global_att_tokens - num_query_tokens) 
    
        """
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        sz = seq_len + num_global_att_tokens
        mask = torch.zeros(num_query_tokens, sz) == 0
        mask[:,train_size:].zero_()
        mask[:,train_size:] |= torch.eye(num_query_tokens) == 1
        return bool_mask_to_att_mask(mask)
        
    @staticmethod
    def generate_global_att_trainset_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        r"""Directs attention between the trainset: essentially fully connected

        Args:
            num_global_att_tokens (int): 
            seq_len (int): 
            num_query_tokens (int): 

        Returns:
            tensor: (seq_len + num_global_att_tokens - num_query_tokens) x num_global_tokens
        """
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        trainset_size = seq_len - num_query_tokens
        mask = torch.zeros(trainset_size, num_global_att_tokens) == 0
        #mask[:,num_global_att_tokens:].zero_()
        #mask[:,num_global_att_tokens:] |= torch.eye(trainset_size) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_globaltokens_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        mask = torch.zeros(num_global_att_tokens, num_global_att_tokens+seq_len-num_query_tokens) == 0
        return bool_mask_to_att_mask(mask)

    def init_weights(self):
        initrange = 1.
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.init_method is not None:
            self.apply(self.init_method)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        """Forwards the points through the transformer
        
        *This is where we need to insert our inter-feature attention.* 

        Args:
            src (tuple): (categorical features (optional), x, y)
            src_mask (tensor, optional): Input mask. Defaults to None - it will be generated within
            single_eval_pos (int, optional): Determines which data point is evaluated. Defaults to None - last data point

        Returns:
            tensor: output
        """
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 2: # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src # Categorical features, x numerical, and y
        print(f"Size of x_src before encoding:{x_src.size()}")
        print(f"Size of y_src before encoding:{y_src.size()}")
        print(f"Size of style_src before encoding:{style_src}")
        #print(f"and first 10 features of the first datapoint {style_src[0:10]}")
        #print(f"and first 10 features of the first datapoint {style_src[0:10] if style_src is not None else style_src}")
        x_src = self.encoder(x_src) # Numerical encoding of x
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src) # encode y
        style_src = self.style_encoder(style_src).unsqueeze(0) if self.style_encoder else torch.tensor([], device=x_src.device) # Style encode categorical features else empty tensor
        print(f"Size of x_src after encoding:{x_src.size()}")
        print(f"Size of y_src after encoding:{y_src.size()}")
        print(f"Size of style_src after encoding:{style_src.size()}")
        
        ### Don't understand global src ### - It seems global_att_embedding and src_mask are linked somehow!
        global_src = torch.tensor([], device=x_src.device) if self.global_att_embeddings is None else \
            self.global_att_embeddings.weight.unsqueeze(1).repeat(1, x_src.shape[1], 1) 
            
        if src_mask is not None: assert self.global_att_embeddings is None or isinstance(src_mask, tuple)
        if src_mask is None: # this is RUN by default: src_mask=None 
            if self.global_att_embeddings is None: # this is RUN by default: global_att_embeddings=None
                full_len = len(x_src) + len(style_src) 
                if self.full_attention:
                    src_mask = bool_mask_to_att_mask(torch.ones((full_len, full_len), dtype=torch.bool)).to(x_src.device) # Full attention mask is create - f x f tensor with 0.0
                elif self.efficient_eval_masking: # This splits the data set into training and evaluation - mask a single number
                    src_mask = single_eval_pos + len(style_src)
                else:
                    src_mask = self.generate_D_q_matrix(full_len, len(x_src) - single_eval_pos).to(x_src.device) # Creates 
            else:
                src_mask_args = (self.global_att_embeddings.num_embeddings,
                                 len(x_src) + len(style_src),
                                 len(x_src) + len(style_src) - single_eval_pos)
                src_mask = (self.generate_global_att_globaltokens_matrix(*src_mask_args).to(x_src.device),
                            self.generate_global_att_trainset_matrix(*src_mask_args).to(x_src.device),
                            self.generate_global_att_query_matrix(*src_mask_args).to(x_src.device))

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos] # y is added to x training set
        # print(global_src.shape, style_src.shape, train_x.shape, x_src[single_eval_pos:].shape)
        src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output[single_eval_pos+len(style_src)+(self.global_att_embeddings.num_embeddings if self.global_att_embeddings else 0):]

    @torch.no_grad()
    def init_from_small_model(self, small_model):
        assert isinstance(self.decoder, nn.Linear) and isinstance(self.encoder, (nn.Linear, nn.Sequential)) \
               and isinstance(self.y_encoder, (nn.Linear, nn.Sequential))

        def set_encoder_weights(my_encoder, small_model_encoder):
            my_encoder_linear, small_encoder_linear = (my_encoder, small_model_encoder) \
                if isinstance(my_encoder, nn.Linear) else (my_encoder[-1], small_model_encoder[-1])
            small_in_dim = small_encoder_linear.out_features
            my_encoder_linear.weight.zero_()
            my_encoder_linear.bias.zero_()
            my_encoder_linear.weight[:small_in_dim] = small_encoder_linear.weight
            my_encoder_linear.bias[:small_in_dim] = small_encoder_linear.bias

        set_encoder_weights(self.encoder, small_model.encoder)
        set_encoder_weights(self.y_encoder, small_model.y_encoder)

        small_in_dim = small_model.decoder.in_features

        self.decoder.weight[:, :small_in_dim] = small_model.decoder.weight
        self.decoder.bias = small_model.decoder.bias

        for my_layer, small_layer in zip(self.transformer_encoder.layers, small_model.transformer_encoder.layers):
            small_hid_dim = small_layer.linear1.out_features
            my_in_dim = my_layer.linear1.in_features

            # packed along q,k,v order in first dim
            my_in_proj_w = my_layer.self_attn.in_proj_weight
            small_in_proj_w = small_layer.self_attn.in_proj_weight

            my_in_proj_w.view(3, my_in_dim, my_in_dim)[:, :small_in_dim, :small_in_dim] = small_in_proj_w.view(3,
                                                                                                               small_in_dim,
                                                                                                               small_in_dim)
            my_layer.self_attn.in_proj_bias.view(3, my_in_dim)[:,
            :small_in_dim] = small_layer.self_attn.in_proj_bias.view(3, small_in_dim)

            my_layer.self_attn.out_proj.weight[:small_in_dim, :small_in_dim] = small_layer.self_attn.out_proj.weight
            my_layer.self_attn.out_proj.bias[:small_in_dim] = small_layer.self_attn.out_proj.bias

            my_layer.linear1.weight[:small_hid_dim, :small_in_dim] = small_layer.linear1.weight
            my_layer.linear1.bias[:small_hid_dim] = small_layer.linear1.bias

            my_layer.linear2.weight[:small_in_dim, :small_hid_dim] = small_layer.linear2.weight
            my_layer.linear2.bias[:small_in_dim] = small_layer.linear2.bias

            my_layer.norm1.weight[:small_in_dim] = math.sqrt(small_in_dim / my_in_dim) * small_layer.norm1.weight
            my_layer.norm2.weight[:small_in_dim] = math.sqrt(small_in_dim / my_in_dim) * small_layer.norm2.weight

            my_layer.norm1.bias[:small_in_dim] = small_layer.norm1.bias
            my_layer.norm2.bias[:small_in_dim] = small_layer.norm2.bias


class TransformerEncoderDiffInit(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
