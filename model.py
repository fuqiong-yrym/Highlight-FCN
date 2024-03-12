import torch
from torch import nn
def get_fcn_model(pretrained = True, orig = True):
    num_classes = 2
    model = torch.hub.load('pytorch/vision:v0.10.0','fcn_resnet50',pretrained=pretrained)
    if orig:
        return model
    else: # modify fcn's classification head: 21 classes --> 2 classes  
        for params in model.parameters():
          params.requires_grad = False     
        inchannels_of_last_layer = model.classifier[4].in_channels
        inchannels_of_last_layer_aux = model.aux_classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(inchannels_of_last_layer,num_classes,kernel_size=(1,1),stride=(1,1))
        model.aux_classifier[4] = nn.Conv2d(inchannels_of_last_layer_aux,num_classes, kernel_size=(1,1),stride=(1,1))
        total_params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad==True)
        print(f'The number of trainable parameters is: {total_params_trainable}')
        return model


