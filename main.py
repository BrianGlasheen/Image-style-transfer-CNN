from PIL import Image
import numpy as np
import cv2
import torch
import torch.optim as optim
import torchvision.transforms.functional as functional
from torchvision import models

def image_to_tensor(path, max_size = 250, shape = None):
    image = Image.open(path)
    
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape

    image = functional.resize(image, size)
    image = functional.to_tensor(image)
    image = functional.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = image.unsqueeze(0)

    return image


def tensor_to_img(tensor):    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model):
    # and the style representation on layers ‘conv1_1’, ‘conv2_1’, ‘conv3_1’, ‘conv4_1’ and ‘conv5_1
    # all used for style except conv4_2, for content
    layers = {'0': 'conv1_1', 
                '5': 'conv2_1', 
                '10': 'conv3_1', 
                '19': 'conv4_1',
                '21': 'conv4_2',
                '28': 'conv5_1'}
        
    features = {}
    temp = image
    for name, layer in model._modules.items():
        temp = layer(temp)
        if name in layers:
            features[layers[name]] = temp
            
    return features

def gram(tensor):
    # from https://towardsdatascience.com/neural-networks-intuitions-2-dot-product-gram-matrix-and-neural-style-transfer-5d39653e7916
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, -1)

    return torch.mm(tensor, tensor.t())


def transfer_style(content_path, style_path, epochs = 350, lr = .015, max_iterations = 5):
    # # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
    vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features
    for param in vgg.parameters(): 
        param.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available
    vgg.to(device)

    # load images into tensors
    content = image_to_tensor(content_path).to(device)
    style = image_to_tensor(style_path, shape=content.shape[-2:]).to(device) # resize to first image

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # "These feature correlations are given by the Gram matrix Gl ∈ RNl×Nl , where Glij is the inner product between the vectorised feature maps i and j in layer l:"
    # precompute gram matrix for each layer used in style
    gram_style = {layer: gram(style_features[layer]) for layer in style_features} # dic to store layer name : gram matrix

    # make a copy to update during gradient descent
    target = content.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 1,
                    'conv2_1': 0.75,
                    'conv3_1': 0.2,
                    'conv4_1': 0.2,
                    'conv5_1': 0.2}

    # eq 7, content weight is alpha
    #       style weight is beta
    # "The ratio α/β was either 1 × 10−3 (Fig 3 B), 8 × 10−4 (Fig 3C), 5 × 10−3 (Fig 3 D), or 5 × 10−4 (Fig 3 E, F)""
    content_weight = 5
    style_weight = 1e4

    # "Here we use L-BFGS [32], which we found to work best for image synthesis"
    optimizer = optim.LBFGS([target], lr = lr, max_iter = max_iterations) # higher max iterations converges much faster, takes longer, number of 'jumps' l-bfgs uses
    # with more time id like to fully explore optimizer to see how it impacts ouput

    def closure():
        optimizer.zero_grad()
        
        target_features = get_features(target, vgg)
        
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        style_loss = 0
        
        for layer in style_weights:
            target_feature = target_features[layer]

            gram_target = gram(target_feature)
            b, c, h, w = target_feature.shape
            style_gram = gram_style[layer]
            layer_style_loss = style_weights[layer] * torch.mean((gram_target - style_gram)**2)

            style_loss += layer_style_loss / (c * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        
        return total_loss

    # epochs = 200
    content_dir = content_path.split(".")[0]
    style_name = style_path.split(".")[0].split("/")[1]

    for epoch in range(1, epochs + 1):
        optimizer.step(closure)
        print(epoch, "/", epochs)
        
        if epoch % 50 == 0:
            img_cv = cv2.cvtColor((255 * tensor_to_img(target)).astype('uint8'), cv2.COLOR_RGB2BGR)
            # cv2.putText(img_cv, f"epoch {epoch}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
            print(f"saved {content_dir}/{style_name}-{epoch}.png")
            cv2.imwrite(f"{content_dir}/{style_name}-{epoch}.png", img_cv)


# content = "images/tamu.png"
# style = "images/minotaur.png"
# transfer_style(content, style)

content_ims = ["images/tamu.png", "images/tuebingen-neckarfront.png", "images/giraffe.png"]
style_ims = ["images/scream.png", "images/starry-night.png", "images/minotaur.png"]

for content in content_ims:
    for style in style_ims:
        transfer_style(content, style)
