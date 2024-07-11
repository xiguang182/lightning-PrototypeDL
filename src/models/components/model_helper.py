import torch
import os

def makedirs(path: str) -> None:
    """Create a directory if it does not exist.

    :param path: The path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def list_of_norms(x: torch.Tensor) -> torch.Tensor:
    '''
    Given a list of vectors, X = [x_1, ..., x_n], we return a list of norms
    [||x_1||, ..., ||x_n||].
    '''
    return torch.sum(torch.pow(x, 2), dim = 1)
    # return torch.sum(x ** 2, -1)

def list_of_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    
    Two tensors are “broadcastable” if the following rules hold:
    Each tensor has at least one dimension.
    When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
    '''
    # device issue, cannot be automaticcally solved by lightning
    y = y.to(x.device)
    XX = torch.reshape(list_of_norms(x), (-1, 1))
    YY = torch.reshape(list_of_norms(y), (1, -1))
    
    XY = torch.matmul(x, torch.transpose(y, 0, 1))
    # broadcasting to compute the pairwise squared euclidean distances
    output = XX - 2 * XY + YY 
    return output

def print_and_write(file, string) -> None:
    """Print a string to stdout and write it to a file.

    :param file: The file object to write to.
    :param string: The string to print and write.
    """
    print(string)
    file.write(string + "\n")

def log_figure(prototype_imgs,  path, name):
    num_cols = 5
    num_rows = prototype_imgs.shape[0] // num_cols + 1 if prototype_imgs.shape[0] % num_cols != 0 else prototype_imgs.shape[0] // num_cols
    #  todo
    pass