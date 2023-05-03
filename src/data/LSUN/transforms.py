from PIL import Image

class CenterCrop():
    """Crop the image to a square which is centered and has the size equal to the min of height and width.        
    
    """
    def __init__(self) -> None:
        pass
    
    def __call__(self, img):
        """
        Arguments
        ---------
            img (PIL.Image.Image)
        """
        is_pil_image(img)
        
        w, h = img.size
        size = min(h, w)
        left = (w - size) // 2
        top = (h - size) // 2
        right = left + size
        bottom = top + size

        return img.crop((left,top,right,bottom))
    
class ReScale(object):
    """Rescale image by given output size.

    Arguments
    ---------
    - output_size (tuple or int): If tuple, output is matched to the output_size.
        If int, smaller of image edges is matched to output_size keeping aspect
        ratio the same.
    
    """
    def __init__(self, output_size) -> None:
        assert isinstance(output_size, (tuple, int)), "output_size should be tuple or int."
        self.output_size = output_size

    def __call__(self, img):
        """
        Arguments
        ---------
        - img (PIL.Image.Image)
        """
        is_pil_image(img)
        
        w, h = img.size

        if isinstance(self.output_size, int):
            if h > w:
                h_resized, w_resized = h/w*self.output_size, self.output_size
            else:
                h_resized, w_resized = self.output_size, w/h*self.output_size
            
        else:
            h_resized, w_resized = self.output_size

        h_resized, w_resized = int(h_resized), int(w_resized)
        img = img.resize((h_resized, w_resized))
        return img

def is_pil_image(img):
    if not isinstance(img, Image.Image):
        raise TypeError("img should be PIL Image")
