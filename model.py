import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from models.configs import get_b16_config, get_b32_config, get_testing
from models.modeling import VisionTransformer

vit = np.load('models/ViT-B_16.npz')
res = models.resnet50(pretrained=True)

''' Multi-task models for hard and soft sharing parameters. '''

class MultiTaskHS(nn.Module):
    def __init__(self, img_size, hidden_features, class1, class2, model_config):
        super(MultiTaskHS, self).__init__()
        
        # ViT UNfrozen config
        if model_config == 'vit':
            self.backbone = VisionTransformer(get_b16_config(), img_size=img_size, vis=True)
            self.backbone.load_from(vit)
            del self.backbone.head

            ct = 0
            for child in self.backbone.transformer.encoder.layer.children():
                ct += 1
                if ct <= 6:
                    for param in child.parameters():
                        param.requires_grad = False

            self.linear = nn.Linear(768, 768)
            self.linear2 = nn.Linear(768, hidden_features)
        
        # Vit frozen config
        if model_config == 'fr_vit':
            self.backbone = VisionTransformer(get_b16_config(), img_size=img_size, vis=True)
            self.backbone.load_from(vit)
            del self.backbone.head

            for param in self.backbone.parameters():
                param.requires_grad = False

            self.linear = nn.Linear(768, 768)
            self.linear2 = nn.Linear(768, hidden_features)

        # Frozen ResNet50 config
        if model_config == 'fr_res':
            print('frozen resnet training')
            self.backbone = res

            for param in self.backbone.parameters():
                param.requires_grad = False

            self.linear = nn.Linear(1000, 768)
            self.linear2 = nn.Linear(768, hidden_features)

        # UNfrozen ResNet50 config
        if model_config == 'res':
            self.backbone = res

            ct = 0
            for child in self.backbone.children():
                ct += 1
                if ct < 6:
                    for param in child.parameters():
                        param.requires_grad = False

            self.linear = nn.Linear(1000, 768)
            self.linear2 = nn.Linear(768, hidden_features)
    

        self.artist_net = nn.Sequential(nn.Linear(hidden_features, hidden_features // 2),
                    nn.ReLU(), nn.Linear(hidden_features // 2, class1))

        self.date_net = nn.Sequential(nn.Linear(hidden_features, hidden_features // 4),
                    nn.ReLU(), nn.Linear(hidden_features // 4, 1))

        self.era_net = nn.Sequential(nn.Linear(hidden_features, hidden_features // 4),
                    nn.ReLU(), nn.Linear(hidden_features // 4, class2))
    
    def forward(self, x, model_config):

        # ViT forward pass
        if model_config == 'fr_vit' or model_config == 'vit':
            x, attn = self.backbone.transformer(x)
            x = F.relu(x[:, 0])
            x = self.linear(x)
            x = F.relu(x)
            x = self.linear2(x)
            x = F.relu(x)

            return self.artist_net(x), self.date_net(x), self.era_net(x), attn
        
        # ResNet50 forward pass
        if model_config == 'fr_res' or model_config == 'res':
            x = self.backbone(x)
            x = self.linear(x)
            x = F.relu(x)
            x = self.linear2(x)
            x = F.relu(x)

            return self.artist_net(x), self.date_net(x), self.era_net(x), 0


''' Single-task models '''

class SingleTaskClassification(nn.Module):
    def __init__(self, img_size, hidden_features, class1, divisor, model_config):
        super(SingleTaskClassification, self).__init__()

        # ViT UNfrozen config
        if model_config == 'vit':
            self.backbone = VisionTransformer(get_b16_config(), img_size=img_size, vis=True)
            self.backbone.load_from(vit)
            del self.backbone.head

            ct = 0
            for child in self.backbone.transformer.encoder.layer.children():
                ct += 1
                if ct <= 6:
                    for param in child.parameters():
                        param.requires_grad = False

            self.net = nn.Sequential(nn.Linear(768, hidden_features // 2),
                    nn.ReLU(), nn.Linear(hidden_features // 2, hidden_features // divisor),
                    nn.ReLU(), nn.Linear(hidden_features // divisor, class1))
        
        # Vit frozen config
        if model_config == 'fr_vit':
            self.backbone = VisionTransformer(get_b16_config(), img_size=img_size, vis=True)
            self.backbone.load_from(vit)
            del self.backbone.head

            for param in self.backbone.parameters():
                param.requires_grad = False

            self.net = nn.Sequential(nn.Linear(768, hidden_features // 2),
                        nn.ReLU(), nn.Linear(hidden_features // 2, hidden_features // divisor),
                        nn.ReLU(), nn.Linear(hidden_features // divisor, class1))

        # Frozen ResNet50 config
        if model_config == 'fr_res':
            self.backbone = res

            for param in self.backbone.parameters():
                param.requires_grad = False

            self.net = nn.Sequential(nn.Linear(1000, hidden_features // 2),
                        nn.ReLU(), nn.Linear(hidden_features // 2, hidden_features // divisor),
                        nn.ReLU(), nn.Linear(hidden_features // divisor, class1))

        # UNfrozen ResNet50 config
        if model_config == 'res':
            self.backbone = res

            ct = 0
            for child in self.backbone.children():
                ct += 1
                if ct < 6:
                    for param in child.parameters():
                        param.requires_grad = False

            self.net = nn.Sequential(nn.Linear(1000, hidden_features // 2),
                        nn.ReLU(), nn.Linear(hidden_features // 2, hidden_features // divisor),
                        nn.ReLU(), nn.Linear(hidden_features // divisor, class1))


    def forward(self, x, model_config):
        # ViT forward pass
        if model_config == 'fr_vit' or model_config == 'vit':
            x, attn = self.backbone.transformer(x)
            x = F.relu(x[:, 0])
            return self.net(x), attn
        
        # ResNet50 forward pass
        if model_config == 'fr_res' or model_config == 'res':
            x = self.backbone(x)
            x = F.relu(x)
            return self.net(x), 0


class SingleTaskRegression(nn.Module):
    def __init__(self, img_size, hidden_features, model_config):
        super(SingleTaskRegression, self).__init__()

        # ViT UNfrozen config
        if model_config == 'vit':
            self.backbone = VisionTransformer(get_b16_config(), img_size=img_size, vis=True)
            self.backbone.load_from(vit)
            del self.backbone.head

            ct = 0
            for child in self.backbone.transformer.encoder.layer.children():
                ct += 1
                if ct <= 6:
                    for param in child.parameters():
                        param.requires_grad = False

            self.net = nn.Sequential(nn.Linear(768, hidden_features // 2),
                    nn.ReLU(), nn.Linear(hidden_features // 2, hidden_features // 4),
                    nn.ReLU(), nn.Linear(hidden_features // 4, 1))
        
        # Vit frozen config
        if model_config == 'fr_vit':
            self.backbone = VisionTransformer(get_b16_config(), img_size=img_size, vis=True)
            self.backbone.load_from(vit)
            del self.backbone.head

            for param in self.backbone.parameters():
                param.requires_grad = False

            self.net = nn.Sequential(nn.Linear(768, hidden_features // 2),
                        nn.ReLU(), nn.Linear(hidden_features // 2, hidden_features // 4),
                        nn.ReLU(), nn.Linear(hidden_features // 4, 1))

        # Frozen ResNet50 config
        if model_config == 'fr_res':
            self.backbone = res

            for param in self.backbone.parameters():
                param.requires_grad = False

            self.net = nn.Sequential(nn.Linear(1000, hidden_features // 2),
                        nn.ReLU(), nn.Linear(hidden_features // 2, hidden_features // 4),
                        nn.ReLU(), nn.Linear(hidden_features // 4, 1))

        # UNfrozen ResNet50 config
        if model_config == 'res':
            self.backbone = res

            ct = 0
            for child in self.backbone.children():
                ct += 1
                if ct < 6:
                    for param in child.parameters():
                        param.requires_grad = False

            self.net = nn.Sequential(nn.Linear(1000, hidden_features // 2),
                        nn.ReLU(), nn.Linear(hidden_features // 2, hidden_features // 4),
                        nn.ReLU(), nn.Linear(hidden_features // 4, 1))

    def forward(self, x, model_config):
        # ViT forward pass
        if model_config == 'fr_vit' or model_config == 'vit':
            x, attn = self.backbone.transformer(x)
            x = F.relu(x[:, 0])
            return self.net(x), attn
        
        # ResNet50 forward pass
        if model_config == 'fr_res' or model_config == 'res':
            x = self.backbone(x)
            x = F.relu(x)
            return self.net(x), 0