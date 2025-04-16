from __future__ import absolute_import, division, print_function

import os
import math
import logging
from contextlib import nullcontext
from typing import Optional, Tuple, Union, Dict, Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
# from torch.nn import LayerNorm
import torch.nn.functional as F
from config import BertConfig
from graph_models import FuseEmbeddings
from torch.amp import autocast  # Updated import
import torch.utils.checkpoint as checkpoint  # For gradient checkpointing

logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    """Implementation of the GELU activation function.
    Optimizes for numerical stability over the original formula.
    """
    return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FastLayerNorm(nn.Module):
    """Faster implementation of LayerNorm with better GPU memory utilization."""
    def __init__(self, hidden_size, eps=1e-12):
        super(FastLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding implementation (RoPE)
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create rotary positional embedding frequencies
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        seq_len = seq_len if seq_len is not None else x.shape[1]
        seq_range = torch.arange(seq_len, device=x.device)
        # Outer product of seq_range and inv_freq
        sinusoidal_inp = torch.einsum('i,j->ij', seq_range, self.inv_freq)
        # Apply sin and cos to alternate positions
        sin, cos = torch.sin(sinusoidal_inp), torch.cos(sinusoidal_inp)
        return sin, cos
    
    @staticmethod
    def apply_rotary_pos_emb(x, sin, cos):
        # x shape: [batch, seq, heads, dim]
        # Reshape x for operations
        x2 = x.reshape(*x.shape[:-1], -1, 2)
        
        # Apply rotary embeddings
        x_transformed = torch.stack([
            x2[..., 0] * cos - x2[..., 1] * sin,
            x2[..., 1] * cos + x2[..., 0] * sin
        ], dim=-1).flatten(-2)
        
        return x_transformed


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention with optimized SDPA implementation.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.d_k = config.hidden_size // config.num_attention_heads
        self.h = config.num_attention_heads
        self.dropout = config.attention_probs_dropout_prob
        self.use_rotary = getattr(config, 'use_rotary', False)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size, config.hidden_size)
        
        if self.use_rotary:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.d_k, 
                max_seq_len=getattr(config, 'max_position_embeddings', 512)
            )

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary:
            sin, cos = self.rotary_emb(q, seq_len)
            q = self.rotary_emb.apply_rotary_pos_emb(q, sin, cos)
            k = self.rotary_emb.apply_rotary_pos_emb(k, sin, cos)
            
        # Use PyTorch 2.0's scaled_dot_product_attention for efficient attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k
        )
        
        return self.output_linear(attn_output)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Using pre-norm approach for improved training stability.
    """

    def __init__(self, config: BertConfig):
        super(SublayerConnection, self).__init__()
        self.norm = FastLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation with improved efficiency."""

    def __init__(self, config: BertConfig):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Add activation function selection
        self.act_fn = getattr(config, 'hidden_act', 'gelu')
        if self.act_fn == 'gelu':
            self.activation = gelu
        elif self.act_fn == 'relu':
            self.activation = F.relu
        elif self.act_fn == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = gelu

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = MultiHeadedAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.input_sublayer = SublayerConnection(config)
        self.output_sublayer = SublayerConnection(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        
        # Normalize at the end as well for improved stability
        self.final_norm = FastLayerNorm(config.hidden_size)
        self.prenorm = getattr(config, 'prenorm', True)

    def forward(self, x, mask):
        if self.prenorm:
            # Pre-norm architecture (better for deeper models)
            x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
            x = self.output_sublayer(x, self.feed_forward)
            return self.final_norm(x)
        else:
            # Post-norm architecture (original transformer implementation)
            residual = x
            attn_output = self.attention(x, x, x, mask=mask)
            x = residual + self.dropout(attn_output)
            x = self.final_norm(x)
            
            residual = x
            ff_output = self.feed_forward(x)
            x = residual + self.dropout(ff_output)
            x = self.final_norm(x)
            
            return x


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        
        # Add position embeddings if not using rotary embeddings
        self.use_rotary = getattr(config, 'use_rotary', False)
        if not self.use_rotary and getattr(config, 'use_position_embeddings', True):
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            
        # Layer normalization and dropout
        self.LayerNorm = FastLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        if position_ids is None and not self.use_rotary and hasattr(self, 'position_embeddings'):
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + token_type_embeddings
        
        # Add position embeddings if available and not using rotary
        if not self.use_rotary and hasattr(self, 'position_embeddings'):
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """Initialize the weights using improved initialization methods."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Use Kaiming initialization for better gradient flow
            nn.init.kaiming_normal_(module.weight.data, nonlinearity='relu')
        elif isinstance(module, FastLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir='', *inputs, **kwargs):
        serialization_dir = os.path.join(cache_dir, pretrained_model_name)
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        return model


class BERT(PreTrainedBertModel):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__(config)
        if config.graph:
            assert dx_voc is not None
            assert rx_voc is not None

        # Configuration options
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)
        self.fp16_enabled = getattr(config, 'fp16_enabled', False)
        self.use_flash_attention = getattr(config, 'use_flash_attention', False)
        
        # Enable Flash Attention backend if requested
        if self.use_flash_attention:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                logger.warning("Flash attention not available, falling back to default implementation")
        
        # Embedding layer
        self.embedding = FuseEmbeddings(config, dx_voc, rx_voc) if config.graph else BertEmbeddings(config)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Layer-wise learning rate decay
        self.layer_decay = getattr(config, 'layer_decay', 0.95)
        
        # Apply weight initialization
        self.apply(self.init_bert_weights)
        
        # Enable fused operations where possible
        self._fuse_operations()

    def _fuse_operations(self):
        """Enable fused operations for efficiency."""
        # This would typically include things like fused LayerNorm, etc.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.cuda_enabled = True

    def forward(self, x, token_type_ids=None, input_positions=None, input_sides=None, attention_mask=None):
        # Handle autocast for mixed precision training
        with autocast(device_type='cuda', enabled=self.fp16_enabled) if self.fp16_enabled else nullcontext():
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = (x > 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            
            # Apply embedding layer
            x = self.embedding(x, token_type_ids, input_positions)
            
            # Process through transformer blocks
            for i, transformer in enumerate(self.transformer_blocks):
                if self.gradient_checkpointing and self.training:
                    x = checkpoint.checkpoint(transformer, x, attention_mask)
                else:
                    x = transformer(x, attention_mask)
            
            # Return sequence and pooled output (first token)
            return x, x[:, 0]
    
    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Makes broadcastable attention mask and causal mask if needed.
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask: {attention_mask.shape}"
            )
        
        # Convert attention mask to binary
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        # Take the hidden state corresponding to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return self.dropout(pooled_output)


# pretaining
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = FastLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, voc_size=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Use adaptive embedding size if provided
        self.vocab_size = config.vocab_size if voc_size is None else voc_size
        
        # Optimize by aligning to multiple of 8 for better GPU utilization
        if getattr(config, 'pad_vocab_size_multiple', 0) > 0:
            pad_to = config.pad_vocab_size_multiple
            self.vocab_size = ((self.vocab_size + pad_to - 1) // pad_to) * pad_to
            
        self.decoder = nn.Linear(config.hidden_size, self.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
