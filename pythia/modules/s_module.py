import torch
from torch import nn
from torch.nn import functional as F

from pythia.modules.layers import ReLUWithWeightNormFC


class S_GNN(nn.Module):
    def __init__(self, f_engineer, bb_dim, feature_dim, l_dim, inter_dim):
        super(S_GNN, self).__init__()
        self.f_engineer = f_engineer
        self.bb_dim = bb_dim
        self.feature_dim = feature_dim
        self.l_dim = l_dim
        self.inter_dim = inter_dim

        self.bb_proj = ReLUWithWeightNormFC(10, self.bb_dim)
        self.fea_fa1 = ReLUWithWeightNormFC(self.bb_dim + self.feature_dim, self.bb_dim + self.feature_dim)
        self.fea_fa2 = ReLUWithWeightNormFC(self.bb_dim + self.feature_dim, self.bb_dim + self.feature_dim)
        self.fea_fa3 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.fea_fa4 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.fea_fa5 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.l_proj1 = ReLUWithWeightNormFC(self.l_dim, 2 * (self.bb_dim + self.feature_dim))
        self.l_proj2 = ReLUWithWeightNormFC(self.l_dim, 2 * (self.bb_dim + self.feature_dim))
        self.output_proj = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), self.feature_dim)

        # self.expedient = ReLUWithWeightNormFC(2 * self.feature_dim, self.feature_dim // 2)

        # self.l_proj = ReLUWithWeightNormFC(2048, self.feature_dim + self.bb_dim)
        # self.gate = nn.Tanh()
        # self.v_recover = ReLUWithWeightNormFC(self.feature_dim, 2048)

    def reset_parameters(self):
        pass

    def bb_process(self, bb):
        """
        :param bb: [B, num, 4], left, down, upper, right
        :return: [B, num(50 or 100), bb_dim]
        """
        bb_size = (bb[:, :, 2:] - bb[:, :, :2])  # 2
        bb_centre = bb[:, :, :2] + 0.5 * bb_size  # 2
        bb_area = (bb_size[:, :, 0] * bb_size[:, :, 1]).unsqueeze(2)  # 1
        bb_shape = (bb_size[:, :, 0] / (bb_size[:, :, 1] + 1e-14)).unsqueeze(2)  # 1
        return self.bb_proj(torch.cat([bb, bb_size, bb_centre, bb_area, bb_shape], dim=-1))

    def forward(self, l, s, ps, mask_s, it=1):
        """
        # all below should be batched
        :param l: [2048], to guide edge strengths, by attention
        :param s: [50, 300]
        :param ps: [50, 4], same as above
        :param mask_s: int, <num_tokens> <= 50
        :param it: iterations for GNN
        :return: updated s with identical shape
        """
        bb = self.bb_process(ps)  # [B, 50, bb_dim]
        s_with_bb = torch.cat([s, bb], dim=2)  # [B, 50, bb_dim + feature_dim]
        l = l.unsqueeze(1).repeat(1, 50, 1)  # [B,50, l_dim]

        inf_tmp = torch.ones(bb.size(0), 50, 50).to(l.device) * float('-inf')
        mask1 = torch.max(torch.arange(50)[None, :], torch.arange(50)[:, None])
        mask1 = mask1[None, :, :].to(mask_s.device) < mask_s[:, None, None]
        mask2 = torch.arange(50).unsqueeze(1).expand(-1, 50).to(mask_s.device)[None, :, :] >= mask_s[:, None, None]
        inf_tmp[mask1] = 0
        inf_tmp[mask2] = 0
        inf_tmp[torch.eye(50).byte().unsqueeze(0).repeat(bb.size(0), 1, 1)] = float('-inf')
        mask3 = mask_s == 1
        inf_tmp[:, 0, 0][mask3] = 0

        output_mask = (torch.arange(50).to(mask_s.device)[None, :] < \
                       mask_s[:, None]).unsqueeze(2).repeat(1, 1, 2 * self.feature_dim)

        for _ in range(it):
            combined_fea = torch.cat([s_with_bb, self.fea_fa1(s_with_bb) * self.fea_fa2(s_with_bb)],
                                     dim=2)  # [B, 50, 2*(bb_dim + feature_dim)]
            l_masked_source = self.fea_fa3(combined_fea) * self.l_proj1(l)  # [B, 50, 2*(bb_dim + feature_dim)]
            adj = torch.matmul(self.fea_fa4(combined_fea), l_masked_source.transpose(1, 2))  # [B, 50, 50]
            # adj = F.softmax(adj / torch.Tensor([condensed.size(2)]).to(adj.dtype).to(adj.device).sqrt_() + inf_tmp,
            #                 dim=2)  # [B, 100, 100]
            adj = F.softmax(adj + inf_tmp, dim=2)  # [B, 50, 50]
            prepared_source = self.fea_fa5(combined_fea) * self.l_proj2(l)  # [B, 50, 2*(bb_dim + feature_dim)]
            messages = self.output_proj(torch.matmul(adj, prepared_source))  # [B, 50, feature_dim]
            s = torch.cat([s, messages], dim=2)  # [B, 50, 2 * feature_dim]

        return s * output_mask.to(s.dtype), adj


if __name__ == '__main__':
    from torchviz import make_dot

    _i = torch.randn(128, 100, 2048)
    _s = torch.randn(128, 50, 300)
    _pi = torch.randn(128, 100, 4)
    _ps = torch.randn(128, 50, 4)
    _mask_s = torch.randint(0, 50, (128,))
    _it = 2
    module = S_GNN(200, 200)
    result = module(_i, _s, _pi, _ps, _mask_s, _it)
    for res in result:
        print(res.shape)
    make_dot(result, params=dict(module.named_parameters()))
