import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

BATCH_NORM_DECAY  =  1 - 0.9  # pytorch batch norm `momentum  =  1 - counterpart` of tensorflow
BATCH_NORM_EPSILON  =  1e-5
    
def get_act(activation):
    """Only supports ReLU and SiLU/Swish."""
    assert activation in ['relu', 'silu']
    if activation  ==  'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()  # TODO: pytorch's nn.Hardswish() v.s. tf.nn.swish

class GNSiLU(nn.Module):
    """"""

    def __init__(self, out_channels, activation = 'relu', nonlinearity = True, init_zero = False):
        super(GNSiLU, self).__init__()
        self.norm  = nn.GroupNorm(8, out_channels)
        if nonlinearity:
            self.act  =  nn.SiLU()
        else:
            self.act  =  None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out  =  self.norm(input)
        if self.act is not None:
            out  =  self.act(out)
        return out

class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative = True, fold_heads = False):
        super(RelPosSelfAttention, self).__init__()
        self.relative  =  relative
        self.fold_heads  =  fold_heads
        self.rel_emb_w  =  nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h  =  nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std = dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std = dim ** -0.5)

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim  =  q.shape
        q  =  q * (dim ** -0.5)  # scaled dot-product
        logits  =  torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits +=  self.relative_logits(q)
        weights  =  torch.reshape(logits, [-1, heads, h, w, h * w])
        weights  =  F.softmax(weights, dim = -1)
        weights  =  torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out  =  torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out  =  torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w  =  self.relative_logits_1d(q, self.rel_emb_w, transpose_mask = [0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h  =  self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask = [0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim  =  q.shape
        rel_logits  =  torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits  =  torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits  =  self.rel_to_abs(rel_logits)
        rel_logits  =  torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits  =  torch.unsqueeze(rel_logits, dim = 3)
        rel_logits  =  rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits  =  rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _  =  x.shape
        col_pad  =  torch.zeros((bs, heads, length, 1), dtype = x.dtype).to(x.device)
        x  =  torch.cat([x, col_pad], dim = 3)
        flat_x  =  torch.reshape(x, [bs, heads, -1]).to(x.device)
        flat_pad  =  torch.zeros((bs, heads, length - 1), dtype = x.dtype).to(x.device)
        flat_x_padded  =  torch.cat([flat_x, flat_pad], dim = 2)
        final_x  =  torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x  =  final_x[:, :, :length, length - 1:]
        return final_x

class GroupPointWise(nn.Module):
    """"""

    def __init__(self, in_channels, heads = 4, proj_factor = 1, target_dimension = None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels  =  target_dimension // proj_factor
        else:
            proj_channels  =  in_channels // proj_factor
        self.w  =  nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std = 0.01)

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input  =  input.permute(0, 2, 3, 1).float()
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out  =  torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out

class MHSA(nn.Module):


    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type = 'relative', use_pos = True):
        super(MHSA, self).__init__()
        self.q_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)
        self.k_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)
        self.v_proj  =  GroupPointWise(in_channels, heads, proj_factor = 1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type  ==  'relative':
            self.self_attention  =  RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads = True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q  =  self.q_proj(input)
        k  =  self.k_proj(input)
        v  =  self.v_proj(input)

        o  =  self.self_attention(q = q, k = k, v = v)
        return o

class BotBlock(nn.Module):

    def __init__(self, in_dimension, curr_h, curr_w, proj_factor = 2, activation = 'relu', pos_enc_type = 'relative',
                 stride = 1, target_dimension = None, msha_iter = 1):
        super(BotBlock, self).__init__()
        self.w = curr_w
        if stride !=  1 or in_dimension !=  target_dimension:
            self.shortcut  =  nn.Sequential(
                nn.Conv2d(in_dimension, target_dimension, kernel_size = 3, padding = 1, stride = stride),
                GNSiLU(target_dimension, activation = activation, nonlinearity = True),
            )
        else:
            self.shortcut  =  None

        bottleneck_dimension  =  target_dimension // proj_factor
        self.conv1  =  nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size = 3, padding = 1, stride = 1 ),
            GNSiLU(bottleneck_dimension, activation = activation, nonlinearity = True)
        )

        self.mhsa  =  nn.ModuleList()
        for i in range(msha_iter):
            self.mhsa.append( 
                    MHSA(
                    in_channels = bottleneck_dimension, 
                    heads = 4, curr_h = curr_h, curr_w = curr_w,
                    pos_enc_type = pos_enc_type
                )
            )
        conv2_list  =  []
        if stride !=  1:
            assert stride  ==  2, stride
            conv2_list.append(nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2)))  # TODO: 'same' in tf.pooling
        conv2_list.append(GNSiLU(bottleneck_dimension, activation = activation, nonlinearity = True))
        
        self.conv2  =  nn.Sequential(*conv2_list)

        self.conv3  =  nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size = 3,padding = 1, stride = 1),
            GNSiLU(target_dimension, nonlinearity = False, init_zero = True),
        )
        self.last_act  =  get_act(activation)
        self.sigmod = nn.Sigmoid()


    def forward(self, x):
        if self.shortcut is not None:
            shortcut  =  self.shortcut(x)
        else:
            shortcut  =  x
        Q_h  =  Q_w  =  self.w
        N, C, H, W  =  x.shape
        P_h, P_w  =  H // Q_h, W // Q_w
        # print( x.shape)
        x  =  x.reshape(N * P_h * P_w, C, Q_h, Q_w)

        out = self.conv1(x)
        out = self.mhsa[0](out)
        # out_o = out.clone().detach()   
        # out  =  self.mhsa[0](out)
        # out_1  =  self.mhsa[1](out_o)
        # out = self.sigmod (out_1 + out)

        out  =  out.permute(0, 3, 1, 2)  # back to pytorch dim order

        out  =  self.conv2(out)
        out  =  self.conv3(out)

        N1, C1, H1, W1  =  out.shape
        out  =  out.reshape(N, C1, int(H), int(W))

        out +=  shortcut
        out  =  self.last_act(out)

        return out

def _make_bot_layer(ch_in, ch_out, w = 4):

    W  =  H  =  w
    dim_in  =  ch_in
    dim_out  =  ch_out

    stage5  =  []

    stage5.append(
        BotBlock(in_dimension = dim_in, curr_h = H, curr_w = W, stride = 1 , target_dimension = dim_out)
    )

    return nn.Sequential(*stage5)