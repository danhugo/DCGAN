from torchvision.utils import make_grid
import numpy as np
import imageio

def create_grid_image(images, nimage_row: int = 10,  save = False, path_to_save = 'grid_image.png') -> np.ndarray:
    '''Create one single grid images comprised of the batch of images
    
    Parameters
    ----------
    - images (tensor): size B x C x H x W
    - show (bool): default is False to not show created grid image.
        otherwise, show image.
    - nimage_row (int): number of images per row
    - path_to_save (str): path to save the grid image.

    Return
    ------
    - (ndarray): one single image size H x W x C
    '''
    img_grid = make_grid(images, nrow=nimage_row)
    img_grid = 255. * np.transpose(img_grid.detach().cpu().numpy(),(1,2,0))
    img_grid = img_grid.astype(np.uint8)
    if save:
        imageio.imwrite(path_to_save, img_grid)
    return img_grid

def save_gif(path, training_grid_images) -> None:
    '''Save generated grid images to a gif file.

    Parameters
    ----------
    - path (str): path to save file gif.
    - training_grid_images (list): list of gird images. 
        gird image (ndarray): H x W x C, type: uint8
    '''
    imageio.mimsave(path, training_grid_images)