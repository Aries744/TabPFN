import math

import torch
import torch.nn as nn

from tabpfn.utils import normalize_data
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class StyleEncoder(nn.Module):
    r"""Creates a linear layer and encodes the vector passed through.
        So it encodes numerical features (all features at once):
        * takes a batch of datapoints and looks at all features - so looks at datapoint vectors of length = num_hyperparameters
        * encodes (linearly) all datapoint vectors into new vectors of length = em_size
        essentially does the same as class Linear which additionally encodes NaN to 0

        Methods:
        __init__(self, num_hyperparameters, em_size): 
        
            Initiates the Linear layer which encodes the number/vector passed through forward
            
            Args:
                Num_hyperparameters: Number of input layers
                em_size: embedded feature size
            
        forward:
            
            The hyperparameter vector is passed through and the encoded vector is returned.
            
            Args:
                hyperparameters: vector to be linearly encoded.
            Returns: 
                Embedded Vector
    """
    
    def __init__(self, num_hyperparameters, em_size):
        """Initiates numerical embedder

        Args:
            num_hyperparameters (int): Input dimension
            em_size (int): Output dimension
        """
        super().__init__()
        self.em_size = em_size
        self.embedding = nn.Linear(num_hyperparameters, self.em_size)

    def forward(self, hyperparameters):  # Batch x num_hyperparameters
        """Transforms numerical features

        Args:
            hyperparameters (Tensor): Dim(Batch, num_hyperparameters) Input to be embedded/transformed

        Returns:
            Tensor: Linearly embedded (if num_hype = 1) or transformed (if num_hyp > 1)
        """
        
        return self.embedding(hyperparameters)



class StyleEmbEncoder(nn.Module):
    r""" Uses the nn.Embedding function to emmbed the given "hyperparameter"
        So it encodes categorical features (one feature at a time):
        * takes a batch of datapoints and looks at one feature (assert num_hyperparameters == 1)
        * encodes all possible values of this feature into vectors of length = em_size
        * note: max number of distinct values that one feature can get is set to num_embeddings=100 (as I understand)
    
    Methods:
        __init__(self, num_hyperparameters, em_size, num_embeddings=100):
            
            Initiates the embedder from nn.Embedding
            
            Args:

                num_hyperparameters: number of input features to embed
                em_size: Embedding output size
                num_embeddings=100: The number of "classes" - More information nn.Embedding, first parameter.
                
        forward(self, hyperparameters):
            Forwards the features and embeds them according to the nn.Embedder
            
            Args: 
            
                hyperparameters: Input vector to encode
                
            Output:
                Tensor that is passed through 
        """
    def __init__(self, num_hyperparameters, em_size, num_embeddings=100):
        
        r"""
            __init__(self, num_hyperparameters, em_size, num_embeddings=100):
            
            Initiates the embedder from nn.Embedding
            
            Args:

                num_hyperparameters: number of input features to embed
                em_size: Embedding output size
                num_embeddings=100: The number of "classes" of a categorical feature.
                """
                
        super().__init__()
        assert num_hyperparameters == 1
        self.em_size = em_size
        self.embedding = nn.Embedding(num_embeddings, self.em_size)

    def forward(self, hyperparameters):  # Batch x num_hyperparameters
        
        r"""        
        
        forward(self, hyperparameters):
            Forwards the batch of features and embeds them according to the nn.Encoder
            
            Args: 
                hyperparameters: Input vector to encode (Batch x number of features)
                
            Output:
                Embdedded Tensor 
        """
        # Squeeze to remove the numerical features? Not sure
        
        return self.embedding(hyperparameters.squeeze(1))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.device_test_tensor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):# T x B x num_features
        assert self.d_model % x.shape[-1]*2 == 0
        d_per_feature = self.d_model // x.shape[-1]
        pe = torch.zeros(*x.shape, d_per_feature, device=self.device_test_tensor.device)
        #position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        interval_size = 10
        div_term = (1./interval_size) * 2*math.pi*torch.exp(torch.arange(0, d_per_feature, 2, device=self.device_test_tensor.device).float()*math.log(math.sqrt(2)))

        pe[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        return self.dropout(pe).view(x.shape[0],x.shape[1],self.d_model)


Positional = lambda _, emsize: _PositionalEncoding(d_model=emsize)

class EmbeddingEncoder(nn.Module):
        
    """I DO NOT THINK THIS GETS USED later down"""
    
    def __init__(self, num_features, em_size, num_embs=100):
        super().__init__()
        self.num_embs = num_embs
        self.embeddings = nn.Embedding(num_embs * num_features, em_size, max_norm=True)
        self.init_weights(.1)
        self.min_max = (-2,+2)

    @property
    def width(self):
        return self.min_max[1] - self.min_max[0]

    def init_weights(self, initrange):

        r"""
        Initiates the embedding weights.
        
            Args: 
                initrange: The weights are initiated between (-initrange, initrange)
                
            Output:
                The nn.Embedding with randomly allocated weights uniformly sampled between (-initrange, initrange)
                
        """
        
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def discretize(self, x):
        
        """
        Splits (-2, +2) equally, takes the interval and floors the x? Then clamps it between 
        the categorical feature selection between 0 to num_embs - 1?
        """
        
        split_size = self.width / self.num_embs
        return (x - self.min_max[0] // split_size).int().clamp(0, self.num_embs - 1)

    def forward(self, x):  # T x B x num_features
       
        r"""
         Train? x Batch x Number of features
        """
        
        x_idxs = self.discretize(x)
        x_idxs += torch.arange(x.shape[-1], device=x.device).view(1, 1, -1) * self.num_embs
        return self.embeddings(x_idxs).mean(-2)


class Normalize(nn.Module):
    """Normalizes according to mean and std
    
    Args:
        nn.Module: Inherits nn.Module
    """
    
    def __init__(self, mean, std):
        """Initiates the normalizer
        
        Args:
            mean (vector): mean vector
            std (vector): std vector
        """
        
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        """Returns normalized vector

        Args:
            x (Tensor): target

        Returns:
            vector: normalized target vector
        """
        return (x-self.mean)/self.std


def get_normalized_uniform_encoder(encoder_creator):
    """ Returns a function of a that normalizes inputs before hand
    
    This can be used to wrap an encoder that is fed uniform samples in [0,1] and normalizes these to 0 mean and 1 std.
    For example, it can be used as `encoder_creator = get_normalized_uniform_encoder(encoders.Linear)`, now this can
    be initialized with `encoder_creator(feature_dim, in_dim)`.
    :param encoder:
    :return:
    """
    return lambda in_dim, out_dim: nn.Sequential(Normalize(.5, math.sqrt(1/12)), encoder_creator(in_dim, out_dim))


def get_normalized_encoder(encoder_creator, data_std):
    """
    Args:
        encoder_creator (nn.Module): A neural network layer module
        data_std (real): Standard deviation vector

    Returns:
        function: Intakes in_dim and out_dim and gives layers defined by encoder creator whose std is scaled to 1.
    """
    return lambda in_dim, out_dim: nn.Sequential(Normalize(0., data_std), encoder_creator(in_dim, out_dim))


class ZNormalize(nn.Module):
    """Gives Z score of vector x (normalizes to 0 mean / variance 1)

    Args:
        x (tensor):
    """
    def forward(self, x):
        return (x-x.mean(-1,keepdim=True))/x.std(-1,keepdim=True)


class AppendEmbeddingEncoder(nn.Module):
    """Not sure what this is doing? Especially the last line - appending the encoded vector?
    nn.Parameter is unclear to me. Where does num_features gets used I am not sure. emsize Embedding size?
    The tensor gets encoded and appended with some extra features. 
    
    I believe this may be used for appending zeroes when passing through the initial layers.
    """
    def __init__(self, base_encoder, num_features, emsize):
        super().__init__()
        self.num_features = num_features
        self.base_encoder = base_encoder
        self.emb = nn.Parameter(torch.zeros(emsize))

    def forward(self, x):
        if (x[-1] == 1.).all():
            append_embedding = True
        else:
            assert (x[-1] == 0.).all(), "You need to specify as last position whether to append embedding. " \
                                        "If you don't want this behavior, please use the wrapped encoder instead."
            append_embedding = False
        x = x[:-1]
        encoded_x = self.base_encoder(x)
        if append_embedding:
            encoded_x = torch.cat([encoded_x, self.emb[None, None, :].repeat(1, encoded_x.shape[1], 1)], 0)
        return encoded_x

def get_append_embedding_encoder(encoder_creator):
    """Returns the function that allows to increase the dimensionality of the encoded data points

    Args:
        encoder_creator (nn.Module): an Encoder module

    Returns:
        function: a lambda function with input dim and output dim as args.
    """
    return lambda num_features, emsize: AppendEmbeddingEncoder(encoder_creator(num_features, emsize), num_features, emsize)


class VariableNumFeaturesEncoder(nn.Module):
    """Extends the number of features by appending 0s. x is also scaled by:
        
            num_real_features/(num_real_features + num_zero_features)
    """
    def __init__(self, base_encoder, num_features):
        """Initiates the variable numerical features encoder

        Args:
            base_encoder (nn.Module): encoder that takes features
            num_features (_type_): dimensionality intaken by base_encoder
        """
        super().__init__()
        self.base_encoder = base_encoder
        self.num_features = num_features

    def forward(self, x):
        """Scales x, increases dimensionality with 0s and passes through base_encoder

        Args:
            x (tensor): numerical input tensor

        Returns:
            tensor: encoded x_ (where x_ is x appended with 0s)
        """
        x = x * (self.num_features/x.shape[-1])
        x = torch.cat((x, torch.zeros(*x.shape[:-1], self.num_features - x.shape[-1], device=x.device)), -1)
        return self.base_encoder(x)


def get_variable_num_features_encoder(encoder_creator):
    return lambda num_features, emsize: VariableNumFeaturesEncoder(encoder_creator(num_features, emsize), num_features)

class NoMeanEncoder(nn.Module):
    """Removes column mean and encodes.
    
    This can be useful for any prior that is translation invariant in x or y.
    A standard GP for example is translation invariant in x.
    That is, GP(x_test+const,x_train+const,y_train) = GP(x_test,x_train,y_train).
    """
    def __init__(self, base_encoder):
        """Initiates NoMeanEncoder

        Args:
            base_encoder (nn.Module): Encoder Modules
        """
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, x):
        """Takes column mean away and encodes

        Args:
            x (tensor): translation invariant input

        Returns:
            tensor: encoded x - x.mean(0, keepdim=True)
        """
        return self.base_encoder(x - x.mean(0, keepdim=True))


def get_no_mean_encoder(encoder_creator):

    """Outputs function that takes input and output dimension of NoMeanEncoder.

    Args:
        encoder_creator (nn.Module): Encoder Module

    Returns:
        function: intakes input and output dimension of encoder module
    """
    return lambda num_features, emsize: NoMeanEncoder(encoder_creator(num_features, emsize))

Linear = nn.Linear
MLP = lambda num_features, emsize: nn.Sequential(nn.Linear(num_features+1,emsize*2),
                                                 nn.ReLU(),
                                                 nn.Linear(emsize*2,emsize))

class NanHandlingEncoder(nn.Module):
    """" Handling missing values - either adding a 0 if keep_nans = False,
    If keeps_nans = True concatenate num_features dimensions to x, putting -1 for
    NaN, 1 for +inf, 2 for -inf, normalize the column. Encodes some information
    about the relative number of nans in the columns? Not sure why they do thi
    
    Some connection to nan_handling in utils.py - but unclear because that takes care of
    NaN values by itself while this is removing NaN values beforeahand.
    """
    def __init__(self, num_features, emsize, keep_nans=True):
        """Initiates the Linear layer and the parameters

        Args:
            num_features (int): Number of input features
            emsize (int): Number of output features from linear layer
            keep_nans (bool, optional): Augment the input point x2 with normalized NaN load if false
            sets NaN to 0.0. Defaults to True.
        """
        super().__init__()
        self.num_features = 2 * num_features if keep_nans else num_features
        self.emsize = emsize
        self.keep_nans = keep_nans
        self.layer = nn.Linear(self.num_features, self.emsize)

    def forward(self, x):
        """Augments x and passes through linear layer

        Args:
            x (tensor): Input tensor

        Returns:
            tensor: x passed through Linear layer
        """
        if self.keep_nans:
            x = torch.cat([torch.nan_to_num(x, nan=0.0), normalize_data(torch.isnan(x) * -1
                                                          + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * 1
                                                          + torch.logical_and(torch.isinf(x), torch.sign(x) == -1) * 2
                                                          )], -1)
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return self.layer(x)


class Linear(nn.Linear):
    """Linear Layer that sets NaN values to 0.
    """
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        """Intiates linear layer with input num_features and out emsize

        Args:
            num_features (int): Input dimension
            emsize (int): Output dimension
            replace_nan_by_zero (bool, optional): If true sets NaN to 0. Defaults to False.
        """
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        """Passes the x through the linear layer

        Args:
            x (tensor): x to be passed through

        Returns:
            tensor: Linear layer output
        """
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state):
        """Pickling feature
        """
        super().__setstate__(state)
        self.__dict__.setdefault('replace_nan_by_zero', True)


class Conv(nn.Module):
    def __init__(self, input_size, emsize):
        super().__init__()
        self.convs = torch.nn.ModuleList([nn.Conv2d(64 if i else 1, 64, 3) for i in range(5)])
        self.linear = nn.Linear(64,emsize)

    def forward(self, x):
        size = math.isqrt(x.shape[-1])
        assert size*size == x.shape[-1]
        x = x.reshape(*x.shape[:-1], 1, size, size)
        for conv in self.convs:
            if x.shape[-1] < 4:
                break
            x = conv(x)
            x.relu_()
        x = nn.AdaptiveAvgPool2d((1,1))(x).squeeze(-1).squeeze(-1)
        return self.linear(x)


class CanEmb(nn.Embedding):
    def __init__(self, num_features, num_embeddings: int, embedding_dim: int, *args, **kwargs):
        assert embedding_dim % num_features == 0
        embedding_dim = embedding_dim // num_features
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)

    def forward(self, x):
        lx = x.long()
        assert (lx == x).all(), "CanEmb only works with tensors of whole numbers"
        x = super().forward(lx)
        return x.view(*x.shape[:-2], -1)


def get_Canonical(num_classes):
    return lambda num_features, emsize: CanEmb(num_features, num_classes, emsize)


def get_Embedding(num_embs_per_feature=100):
    return lambda num_features, emsize: EmbeddingEncoder(num_features, emsize, num_embs=num_embs_per_feature)
