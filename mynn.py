import torch
from torch import nn
from torch.nn import functional as F


class ConvBnAct(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def huber_loss(bbox_pred, bbox_targets, beta=2.8):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    https://en.wikipedia.org/wiki/Huber_loss
    """
    box_diff = bbox_pred - bbox_targets

    dis_trans = torch.norm(box_diff, dim=1)
    # dis_trans = np.linalg.norm(box_diff.data.cpu().numpy(), axis=1)
    # we also add a metric for dist<2.8 metres.
    inbox_idx = dis_trans <= 2.8
    outbox_idx = dis_trans > 2.8

    # bbox_inside_weights = torch.from_numpy(inbox_idx.astype('float32')).cuda()
    # bbox_outside_weights = torch.from_numpy(outbox_idx.astype('float32')).cuda()
    bbox_inside_weights = inbox_idx.float()
    bbox_outside_weights = outbox_idx.float()

    in_box_pow_diff = 0.5 * torch.pow(box_diff, 2) / beta
    in_box_loss = in_box_pow_diff.sum(dim=1) * bbox_inside_weights

    out_box_abs_diff = torch.abs(box_diff)
    out_box_loss = (out_box_abs_diff.sum(dim=1) - beta / 2) * bbox_outside_weights

    loss_box = in_box_loss + out_box_loss
    N = loss_box.size(0)  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


def dist_to_coord(distance_scaled, bbox):
    z_outputs = distance_scaled * torch.sqrt(1 / torch.reshape(1 + torch.pow(bbox[:, 0], 2) + torch.pow(bbox[:, 1], 2), (-1, 1)))
    x_outputs = torch.reshape(bbox[:, 0], (-1, 1)) * z_outputs
    y_outputs = torch.reshape(bbox[:, 1], (-1, 1)) * z_outputs
    return torch.cat((x_outputs, y_outputs, z_outputs), dim=-1)


def mean_distance(translation_pred, translation):
    distance = torch.sqrt(torch.pow((translation_pred - translation), 2).sum(dim=1)).mean()
    return distance
