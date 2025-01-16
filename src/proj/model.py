import timm

def create_model(architecture="resnet50.a1_in1k", pretrained=False, num_classes=257):
    
    model = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)

    return model
