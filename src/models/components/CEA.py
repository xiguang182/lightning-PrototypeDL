import os
import torch
from torch import nn

from src.utils.proto_utils import list_of_distances

"""
Adopted from @author Oscar Li

Source: https://github.com/OscarcarLi/PrototypeDL
"""

class ConvLayer(nn.Module):
    """A convolutional layer for the CEA model."""

    def __init__(
            self, 
            in_channels: int = 1, 
            out_channels: int = 10,
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
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        self.in_dim = x.shape[-2:]
        return self.activation(self.conv(x))
    
class DeConvLayer(nn.Module):
    """A deconvolutional layer for the CEA model."""

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
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
        self.out_shape = out_shape
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, output_padding=output_padding
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        out_conv = self.deconv(x)
        # trim the output to the desired shape. take the last (shape) rows and columns
        # transpose convolution should be called flip convolution. Because the kernel is flipped.
        if (out_conv.shape[-2:][0] != self.out_shape[0]) & (out_conv.shape[-2:][1] != self.out_shape[1]):
            out_conv = out_conv[:,:,(out_conv.shape[-2:][0] - self.out_shape[0]):,(out_conv.shape[-2:][1] - self.out_shape[1]):]
        return self.activation(out_conv)

class LinearLayer(nn.Module):
    """A linear layer for the CEA model."""

    def __init__(
            self, 
            in_features: int = 784, 
            out_features: int = 10,
            activation = nn.ReLU,
        ) -> None:
        """Initialize a `LinearLayer` module.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param activation: The activation function.
        """
        super().__init__()
        self.activation = activation()
        self.linear = nn.Linear(in_features, out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.activation(self.linear(x))
\
# maybe unnecessary
class SoftmaxLayer(nn.Module):
    """A softmax layer for the CEA model."""

    def __init__(self) -> None:
        """Initialize a `SoftmaxLayer` module.
        """
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.softmax(x)

class EncoderLayer(nn.Module):
    """An encoder layer for the CEA model."""

    def __init__(
            self, 
            in_channels: int = 1, 
            out_channels: int = 10,
            num_layers: int = 4,
            num_maps: int = 32, 
        ) -> None:
        """Initialize a `EncoderLayer` module.
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param num_layers: The number of layers.
        :param num_maps: The number of feature maps. or number of Kernels, out_channels for conv2d
        """
        super().__init__()

        encoder_layers = [ConvLayer(in_channels, num_maps, stride=2, padding=1)]
        for _ in range(num_layers - 2):
            encoder_layers += [ConvLayer(num_maps, num_maps, stride=2, padding=1)]
        encoder_layers += [ConvLayer(num_maps, out_channels, stride=2, padding=1)]
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.encoder(x)

class DecoderLayer(nn.Module):
    def __init__(
            self, 
            in_channels: int = 10, 
            out_channels: int = 1,
            num_layers: int = 4,
            num_maps: int = 32, 
            out_shapes = []
        ) -> None:
        """Initialize a `DecoderLayer` module.
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param num_layers: The number of layers.
        :param num_maps: The number of feature maps. or number of Kernels, out_channels for conv2d
        """
        super().__init__()

        decoder_layers = [DeConvLayer(in_channels, num_maps, out_shapes[-1], stride=2, padding=1, output_padding=1)]
        for i in range(2, num_layers):
            decoder_layers += [DeConvLayer(num_maps, num_maps, out_shapes[-i], stride=2, padding=1, output_padding=1)]
        decoder_layers += [DeConvLayer(num_maps, out_channels, out_shapes[-num_layers], stride=2, padding=1, output_padding=1, activation=nn.Sigmoid)]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.decoder(x)

class PrototypeLayer(nn.Module):
    def __init__(
            self,
            in_channels: int = 10,
            num_prototypes: int = 15, 
        ):
        super().__init__()

        self.prototypes = torch.rand((num_prototypes, in_channels), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return list_of_distances(x, self.prototypes)

class CAEModel(nn.Module):
    def __init__(
            self,
            in_shape: list = [1, 1, 28, 28],
            num_classes: int = 10,
            num_layers: int = 4,
            num_maps: int = 32,
            num_prototypes: int = 15,
        ):
        """
        Initialize a `CAEModel` module.
        :param in_shape: The shape of the input tensor.
        :param num_classes: The number of classes.
        :param num_layers: The number of layers.
        :param num_maps: The number of feature maps. or number of Kernels, out_channels for conv2d
        :param num_prototypes: The number of prototypes.
        """

        super().__init__()

        # Encoder layer
        self.encoder = EncoderLayer(in_channels= in_shape[1], out_channels = num_classes, num_layers = num_layers, num_maps = num_maps)
        
        # prototype layer
        # the number of input channels to the prototype layer is the dimension/length of the encoder output 28-14-7-4-2, (2,2)
        self.in_channels_prototype = self.encoder.forward(torch.randn(in_shape)).view(-1,1).shape[0]
        # 40
        # print(self.in_channels_prototype)

        self.prototype_layer = PrototypeLayer(in_channels = self.in_channels_prototype, num_prototypes = num_prototypes)

        # Decoder layer
        decoder_out_shapes = []
        for layer in self.encoder.modules():
            if isinstance(layer, ConvLayer):
                # print(layer)
                decoder_out_shapes += [list(layer.in_dim)]

        # [[28, 28], [14, 14], [7, 7], [4, 4]]
        # print(decoder_out_shapes)
        self.decoder = DecoderLayer(in_channels = num_classes, out_channels = in_shape[1], num_layers = num_layers, num_maps = num_maps, out_shapes = decoder_out_shapes)
        
        # output layer
        self.fc = nn.Linear(in_features = num_prototypes, out_features = num_classes)
        self.feature_vectors = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: The output tensor.
        """
        encoder_out = self.encoder(x)
        self.feature_vectors = encoder_out
        prototype_out = self.prototype_layer(encoder_out.view(-1, self.in_channels_prototype))
        fc_out = self.fc(prototype_out)
        return fc_out

if __name__ == "__main__":
    _ = CAEModel()
