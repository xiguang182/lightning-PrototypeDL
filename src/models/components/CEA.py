import os
import torch
from torch import nn

"""
Adopted from @author Oscar Li

Source: https://github.com/OscarcarLi/PrototypeDL
"""

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
    XX = torch.reshape(list_of_norms(x), (-1, 1))
    YY = torch.reshape(list_of_norms(y), (1, -1))
    # broadcasting to compute the pairwise squared euclidean distances
    output = XX - 2 * torch.matmul(x, torch.transpose(y, 0, 1)) + YY 
    return output

def print_and_write(file, string) -> None:
    """Print a string to stdout and write it to a file.

    :param file: The file object to write to.
    :param string: The string to print and write.
    """
    print(string)
    file.write(string + "\n")

class ConvLayer(nn.Module):
    """A convolutional layer for the CEA model."""

    def __init__(
            self, 
            input_channels: int = 1, 
            output_channels: int = 10,
            stride = 2,
            padding = 1,
            activation = nn.ReLU,
        ) -> None:
        """Initialize a `ConvLayer` module.

        :param input_channels: The number of input channels.
        :param output_channels: The number of output channels.
        :param stride: The stride of the convolution.
        :param padding: The padding of the convolution.
        """
        super().__init__()

        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=padding
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """

        return self.activation(self.conv(x))
    
class DeConvLayer(nn.Module):
    """A deconvolutional layer for the CEA model."""

    def __init__(
            self, 
            input_channels: int, 
            output_channels: int,
            out_shape,
            stride = 2,
            padding = 1,
            output_padding = 1,
            activation = nn.ReLU,
        ) -> None:
        """Initialize a `DeConvLayer` module.

        :param input_channels: The number of input channels.
        :param output_channels: The number of output channels.
        :param stride: The stride of the convolution.
        :param padding: The padding of the convolution.
        """
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, output_padding=output_padding
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        out_conv = self.deconv(x)
        # trim the output to the desired shape. take the last (shape) rows and columns
        # transpose convolution should be called rotated convolution (to-do: elaborate more.)
        if (out_conv.shape[-2:][0] != self.out_shape[0]) & (out_conv.shape[-2:][1] != self.out_shape[1]):
            out_conv = out_conv[:,:,(out_conv.shape[-2:][0] - self.out_shape[0]):,(out_conv.shape[-2:][1] - self.out_shape[1]):]
        return self.activation(out_conv)



class EncoderLayer(nn.Module):
    """An encoder layer for the CEA model."""

    def __init__(
            self, 
            input_channels: int = 1, 
            output_channels: int = 10,
            num_layers: int = 4,
            num_maps: int = 32, 
        ) -> None:
        """Initialize an `EncoderLayer` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features.
        """
        super().__init__()

        encoder_layers = []

        # encoder_layers += []

        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.model(x)

class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
