"""
Shared Layers - all assume 5D inputs
"""
from .utils import *
from timm.models.layers import DropPath

import torch
from torch import nn
# from mamba_ssm import Mamba
from torch.cuda.amp import autocast

"""
Basic Layers
"""
class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


class FeatureGrid(nn.Module):
    """
    The module for storing learnable embedding.
    """
    def __init__(self, shape, init_scale):
        super().__init__()
        self.register_parameter(name='weight', param=nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=True))
        torch.nn.init.uniform_(self.weight, -init_scale, init_scale)

    def forward(self):
        return self.weight #.clone()


class FeatureBuffer(nn.Module):
    """
    The module for storing non-learnable embedding, e.g. for autoencoder.
    """
    def __init__(self, shape):
        super().__init__()
        self.register_parameter(name='weight', param=nn.Parameter(torch.zeros(shape), requires_grad=False))

    def forward(self, idx, x):
        if x is not None:
            output = self.weight[idx] = x.detach().to(self.weight.dtype)
        else:
            output = self.weight[idx]
        return output


class Conv2d(nn.Conv2d):
    def forward(self, input):
        N, T, H, W, _ = input.shape
        x = input.view(N * T, H, W, -1).permute(0, 3, 1, 2)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = x.permute(0, 2, 3, 1).view(N, T, H, W, -1)
        return x


class Conv3d(nn.Conv3d):
    def forward(self, input):
        x = input.permute(0, 4, 1, 2, 3)
        x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = x.permute(0, 2, 3, 4, 1)
        return x


"""
Advanced Layers
"""
class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 act='gelu', norm='layernorm', bias: bool = True,
                 norm_first=False) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = get_norm(norm)(in_features if self.norm_first else out_features)
        self.act = get_activation(act)()

    def forward(self, input):
        x = self.linear(self.norm(input)) if self.norm_first else self.norm(self.linear(input))
        x = self.act(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding='same', bias=bias)
        self.norm = get_norm(norm)(in_features if self.norm_first else out_features)
        self.act = get_activation(act)()

    def forward(self, input):
        x = self.conv(self.norm(input)) if self.norm_first else self.norm(self.conv(input))
        x = self.act(x)
        return x


class BlockBase(nn.Module):
    def __init__(self, in_features, out_features, layerscale_init, droppath):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layerscale_init = layerscale_init

        # layer scale
        if self.in_features == self.out_features and layerscale_init > 0.:
            self.layerscale = torch.nn.Parameter(self.layerscale_init * torch.ones([self.out_features]), requires_grad=True)
        else:
            self.layerscale = None

        # Stochastic Depth
        if self.in_features == self.out_features and droppath > 0.:
            self.droppath = DropPath(droppath)
        else:
            self.droppath = None

    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}, layerscale_init={layerscale_init}'
        return s.format(**self.__dict__)

    def block_forward(self, input):
        raise NotImplementedError

    def forward(self, input, mask=None):
        x = self.block_forward(input)

        if self.layerscale is not None:
            x = self.layerscale * x

        if self.droppath is not None:
            x = self.droppath(x)

        if mask is not None:
            x = mask * x

        if self.in_features == self.out_features:
            x = x + input

        return x


class MLPBlock(BlockBase):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        self.norm = get_norm(norm)(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def block_forward(self, input):
        x = self.norm(input)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConvNeXtBlock(BlockBase):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        # self.out_features = out_features
        # self.in_features = in_features
        # self.hidden_features = hidden_features
        self.dconv = Conv2d(in_features, in_features, kernel_size, groups=in_features, padding='same', bias=bias)
        self.norm = get_norm(norm)(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def block_forward(self, input):
        # print("in_features", self.in_features, "out_features", self.out_features, "input.shape", input.shape)
        # print("hidden_features.shape", self.hidden_features)
        # print("input.device", input.device)
        x = self.dconv(input)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        # print("x.device",x.device)
        # print("ouput.shape:", x.shape)
        return x


class ConvNeXtBlockLessNorm(BlockBase):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, kernel_size=3, 
                 act='gelu', norm='layernorm', bias: bool = True,
                 layerscale_init=0., dropout=0., droppath=0.):
        super().__init__(in_features, out_features, layerscale_init, droppath)
        self.dconv = Conv2d(in_features, in_features, kernel_size, groups=in_features, padding='same', bias=bias)
        self.norm = get_norm(norm if in_features == out_features else 'none')(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = get_activation(act)()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout2 = nn.Dropout(dropout)
    
    def block_forward(self, input):
        x = self.dconv(input)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class MambaBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 bias: bool = True, dropout=0., act='gelu', norm='layernorm',
                 dim = 1, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        # print(f"MambaLayer: dim: {dim}")
        self.out_features = out_features
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.dim = dim
        self.norm = nn.LayerNorm(dim).cuda()
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.mamba = self.mamba.cuda()
        self.channel_token = channel_token ## whether to use channel as tokens

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias).cuda()
        self.act = get_activation(act)().cuda()
        self.dropout1 = nn.Dropout(dropout).cuda()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias).cuda()
        self.dropout2 = nn.Dropout(dropout).cuda()

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        # assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_norm = x_norm.to('cuda')
        # print("x_norm.is_cuda()", x_norm.device)
        # print("x.shape", x.shape, "x_norm", x_norm.shape)
        if x_norm.shape[1] < 72577000:
            x_mamba = self.mamba(x_norm)
        else:
            split = x_norm.shape[1] // 20
            x_mamba_list = []
            for i in range(20):
                x_norm_i = x_norm[:,split*i:split*(i+1), :]
                x_mamba_i = self.mamba(x_norm_i)
                x_mamba_list.append(x_mamba_i)
            x_mamba = torch.cat(x_mamba_list, dim=1)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        # print("out", out.shape)

        del x, x_norm, x_mamba, x_flat, img_dims, n_tokens, B, d_model

        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        # assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        # assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_norm = x_norm.to('cuda')
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    # @autocast(enabled=False)
    def forward(self, x, mask=None):
        # print("input.device", x.device)
        device = x.device
        x = x.cuda()
        # if x.dtype == torch.float16:
        #     x = x.type(torch.float32)
        if self.channel_token:
            x = self.forward_channel_token(x)
        else:
            x = self.forward_patch_token(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x) 

        x = x.to(device)
        # print("output.device", x.device)
        # print("output.shape", x.shape, "out_features", self.out_features)
        return x


class MambaVisionBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        i = 0
        depths=[2]

        window_size=[8]
        mlp_ratio=4
        qkv_bias=True
        qk_scale=None
        drop_path_rate=0.2
        drop_rate=0.
        attn_drop_rate=0.
        layer_scale=None
        layer_scale_conv=None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        in_dim = out_features//2
        self.in_features = in_features
        self.hidden_features = in_dim
        self.out_features = out_features

        if out_features % 2 != 0:
            dim = out_features + 1
        else:
            dim = out_features

        if dim % 20 == 0:
            num_heads=[dim//20]
        if dim % 10 == 0:
            num_heads=[dim//10]
        elif dim % 7 == 0:
            num_heads=[dim//7]
        elif dim % 3 == 0:
            num_heads=[dim//3]
        
        self.patch_embed = PatchEmbed(in_chans=in_features, in_dim=in_dim, dim=dim)
        self.patch_embed = self.patch_embed.cuda()

        self.level = MambaVisionLayer(dim=dim,
                                    depth=depths[i],
                                    num_heads=num_heads[i],
                                    window_size=window_size[i],
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                    downsample=False,
                                    layer_scale=layer_scale,
                                    layer_scale_conv=layer_scale_conv,
                                    # transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                    transformer_blocks=[],
                                    )
        self.level = self.level.cuda()
        
        
    def forward(self, x, mask=None):
        # print("in_features:", self.in_features, "out_features:", self.out_features, "hidden_features:", self.hidden_features)
        # print("x.shape", x.shape)  #torch.Size([1, 1, 90, 160, 280])
        device = x.device
        x = x.squeeze(1)
        x = x.permute(0,3,1,2)
        x = x.cuda()

        x = self.patch_embed(x)
        # print("patch_embed:", x.shape)
        x = self.level(x)
        # print("level:", x.shape)

        if self.out_features % 2 != 0:
            x = x[:, :-1, :,:]

        x = x.permute(0,2,3,1)
        x = x.unsqueeze(1)
        # print("output.shape", x.shape) 

        x = x.to(device)
        return x
    
class GridTrilinear3D(nn.Module):
    """
    The module for mapping feature maps to a fixed size with trilinear interpolation.
    """
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        #x = F.interpolate(x, size=self.output_size, mode='trilinear', align_corners=self.align_corners)
        T, H, W, C = x.shape
        assert H == self.output_size[1] and W == self.output_size[2], 'F.interpolate has incorrect results in some cases, so use only temporal scale'
        x = x.view(1, 1, T, H * W * C)
        x = F.interpolate(x, size=(self.output_size[0], H * W * C), mode='bilinear', align_corners=self.align_corners)
        x = x.view(self.output_size + (C,))
        return x


"""
Utils
"""
def get_norm(norm, **kwargs):
    if norm == "none":
        return nn.Identity
    elif norm == "layernorm":
        return partial(nn.LayerNorm, eps=1e-6, **kwargs)
    elif norm == "layernorm-no-affine":
        return partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6, **kwargs)
    else:
        raise NotImplementedError


def get_activation(activation):
    if activation == "none":
        return nn.Identity
    elif activation == "relu":
        return nn.ReLU
    elif activation == "relu6":
        return nn.ReLU6
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "gelu_fast":
        return partial(nn.GELU, approximate='tanh')
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "softplus":
        return nn.Softplus
    elif activation == "sin":
        return Sin
    else:
        raise NotImplementedError


def get_block(type, **kwargs):
    if type == 'identity':
        return torch.nn.Identity()
    elif type == 'linear_stem':
        return LinearBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=False, bias=kwargs['bias'])
    elif type == 'conv_stem':
        return Conv2dBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], kernel_size=kwargs['kernel_size'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=False, bias=kwargs['bias'])
    elif type == 'linear_head':
        return LinearBlock(in_features=kwargs['C1'], out_features=kwargs['C2'],
                           act=kwargs['act'], norm=kwargs['norm'], norm_first=True, bias=kwargs['bias'])
    elif type == 'mlp':
        return MLPBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                        act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                        layerscale_init=kwargs['layerscale'],
                        dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    elif type == 'convnext':
        return ConvNeXtBlock(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                             kernel_size=kwargs['kernel_size'], act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                             layerscale_init=kwargs['layerscale'],
                             dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    elif type == 'convnext-lessnorm':
        return ConvNeXtBlockLessNorm(in_features=kwargs['C1'], out_features=kwargs['C2'], hidden_features=kwargs['Ch'],
                                     kernel_size=kwargs['kernel_size'], act=kwargs['act'], norm=kwargs['norm'], bias=kwargs['bias'],
                                     layerscale_init=kwargs['layerscale'],
                                     dropout=kwargs['dropout'], droppath=kwargs['droppath'])
    elif type == "mamba_block":
        return MambaVisionBlock(in_features=kwargs['C1'], out_features=kwargs['C2'])
    else:
        raise NotImplementedError