import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



# Used article https://samuelabiodun.medium.com/how-to-deploy-a-pytorch-model-on-sagemaker-aa9a38a277b6

def model_fn(model_dir):
    logger.info(f"Inside model function loading the model from {model_dir}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)

    # freeze the CNN layers
    for param in model.parameters():
        param.require_grad = False
    
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
​
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    logger.info('Done loading model')
    return model


def input_fn(request_body, content_type='image/jpeg'):
    logger.info('Deserializing the input data.')
    if content_type == 'image/jpeg':
        
        image_data = Image.open(io.BytesIO(request_body))
        
        image_transform = transforms.Compose([
            transforms.Resize(224, 224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return image_transform(image_data)
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')
    
# inference
def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(input_data)
        preds = torch.exp(out)
    return preds
