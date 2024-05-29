import kornia.augmentation as K

def UEraser(input):
    aug = K.AugmentationSequential(
                K.RandomPlasmaBrightness(roughness=(0.1, 0.7), intensity=(0.0, 1.0),
                                         same_on_batch=False, p=1.0, keepdim=True),
                K.RandomPlasmaContrast(roughness=(0.1, 0.7), p=1.0),
                K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=True),
                K.auto.TrivialAugment())
    output = aug(input)
    return output
