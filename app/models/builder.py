import torchvision
from torch import nn

def build_model(name="densenet121", num_classes=1, pretrained=True):
    name = name.lower()
    if name.startswith("resnet"):
        m = getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name.startswith("densenet"):
        m = getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name.startswith("efficientnet"):
        m = getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(name)
    return m
