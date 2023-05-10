import torch
from torch import nn

class GuidedBackprop():
    def __init__(self, model):
        """Guided Backprop to visualize learned features from specific neurons.
        - model: here is Discriminator model
        """
        self.model = model
        self.image_reconstruction = None
        self.model.eval()
        self.last_layer_fmap = None
        self.fmaps = []

    def register_hooks(self, last_layer: str = None):
        '''register hook function for module till specified last layer.
        - last_layer (str):last layer to caculate forward pass and backpropagation pass (default: None).
        '''
        def first_layer_hook(module, grad_out, grad_in):
            '''Get grad out of last layer as reconstructed image.'''
            self.image_reconstruction = grad_out[0] 

        def last_layer_forward_hook(module, input, output):
            self.last_layer_fmap = output[0]
        
        def relu_forward_hook(module, input, output):
            # input, output (tupple) with input is feature input and output is feature output of a module.
            self.fmaps.append(input[0])

        def relu_backward_hook(module, grad_out, grad_in):
            # ReLU is refered as a special case of LeakyReLU with slope = 0
            # In backward pass, ReLU forces grad values at positions where input in forward pass is negative to zero (slope = 0) 
            # With LeakyReLU, these grad values will be turned to slope value.
            prev_fmap = self.fmaps.pop()
            grad = torch.zeros_like(prev_fmap)
            grad[prev_fmap > 0] = 1
            new_grad_out = torch.clamp(grad_in[0], min=0.0) * grad
            return (new_grad_out,)

        modules = list(self.model.named_children())

        # Attach hook to forward and backward pass with paticular module
        for name, module in modules:
            if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU):
                module.register_forward_hook(relu_forward_hook)
                module.register_backward_hook(relu_backward_hook)
            if last_layer is not None and name == last_layer:
                module.register_forward_hook(last_layer_forward_hook)
                break

        # register backward hook for the first conv layer
        first_layer = modules[0][1] 
        first_layer.register_backward_hook(first_layer_hook)

    def generate_reconstructed_image(self, input_image, neuron: int = 0):
        '''Generate reconstructed image follow guided backpropagation process. 
        Arguments
        - input_image (Tensor): C x H x W image
        - neuron (int): used in case to see learned feature of a defined last layer.
        Return
        - np.ndarray: image size H x W x C.
        '''
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        if self.last_layer_fmap is not None:
            grad_target_map = torch.zeros_like(self.last_layer_fmap)
            grad_target_map.view(-1)[neuron] = torch.ones(1)
            self.last_layer_fmap.backward(grad_target_map)
        else:
            grad_target_map = torch.tensor(1)
            model_output.backward(grad_target_map)
        
        result = self.image_reconstruction.data[0]
        return normalize(result)
    
def normalize(image):
    norm = (image - image.mean())/(image.std() + 1e-5)
    norm = norm * 0.1
    norm = norm + 0.5
    norm = norm.clip(0, 1)
    return norm