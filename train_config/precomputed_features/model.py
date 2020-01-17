import torch

from torch import nn
from torchvision import models
from mynn import ConvBnAct

RAD_POWER = 6


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet101(pretrained=True)

        self.local_neck = nn.Sequential(
            ConvBnAct(2048, 512),
            ConvBnAct(512, 256),
            ConvBnAct(256, 64),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1024),  # 4
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),  # 6
            nn.LeakyReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.local_neck[4].weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.local_neck[4].bias, 0)
        nn.init.kaiming_normal_(self.local_neck[6].weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.local_neck[6].bias, 0)

        self.cls_head = nn.Linear(1024, 34)
        nn.init.kaiming_normal_(self.cls_head.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.cls_head.bias, 0)

        self.rot_head = nn.Linear(1024, 4)
        nn.init.kaiming_normal_(self.rot_head.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.rot_head.bias, 0)

        self.trans_ass = nn.Sequential(
            nn.Linear(RAD_POWER, 1, bias=False)    
        )
        nn.init.xavier_normal_(self.trans_ass[0].weight)

        self.trans_neck = nn.Sequential(
             nn.Linear(1024, 128),
             nn.LeakyReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.trans_neck[0].weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.trans_neck[0].bias, 0)

        self.size_neck = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.size_neck[0].weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.size_neck[0].bias, 0)
        nn.init.kaiming_normal_(self.size_neck[2].weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.size_neck[2].bias, 0)
        nn.init.kaiming_normal_(self.size_neck[4].weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.size_neck[4].bias, 0)

        self.trans_head = nn.Linear(256, 1)
        nn.init.xavier_normal_(self.trans_head.weight)
        nn.init.constant_(self.trans_head.bias, 0)

    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x

    def forward(self, x, bbox):
        features = self.extract_features(x)
        ln = self.local_neck(features)
        tn = self.trans_neck(ln)
        rad = torch.sqrt(torch.pow(bbox[:, 0:1], 2) + torch.pow(bbox[:, 1:2], 2))
        rad_powers = rad
        for i in range(1, RAD_POWER):
            rad_powers = torch.cat((rad_powers, torch.pow(rad, i + 1)), dim=-1)

        ta = 1 + self.trans_ass(rad_powers)

        sn = self.size_neck(bbox[:, 2:])

        distance = self.trans_head(torch.cat((sn, tn), dim=-1))

        distance_scaled = distance * ta

        cls_score = self.cls_head(ln)
        rotation = self.rot_head(ln)

        return dict(cls_score=cls_score, rotation=rotation, distance=distance_scaled)


def prepare_batch(batch):
    for k in batch:
        if k == 'label':
            batch[k] = batch[k].long()
        else:
            batch[k] = batch[k].float()
        batch[k] = batch[k].cuda()
    x = dict(x=batch['image'], bbox=batch['bbox'])
    y = batch
    return x, y
