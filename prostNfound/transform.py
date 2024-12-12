
class RandomTranslation: 
    def __init__(self, translation=(0.2, 0.2)): 
        self.translation = translation

    def __call__(self, *images):
        from torchvision.transforms.functional import affine
        from random import uniform

        h_factor, w_factor = uniform(-self.translation[0], self.translation[0]), uniform(-self.translation[1], self.translation[1])

        outputs = []
        for image in images:
            H, W = image.shape[-2:]
            translate_x = int(w_factor * W)
            translate_y = int(h_factor * H)
            outputs.append(affine(image, angle=0, translate=(translate_x, translate_y), scale=1, shear=0))

        return outputs[0] if len(outputs) == 1 else outputs