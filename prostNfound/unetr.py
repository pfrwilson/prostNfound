import torch
from monai.networks.nets.unetr import (
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)


class UNETR(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        embedding_size=768,
        feature_size=64,
        out_channels=1,
        norm_name="instance",
        input_size=1024, 
        output_size=256,
    ):
        """
        Args: 
            image_encoder: a vision transformer backbone model. Its call pattern should be `image_encoder(x, return_hiddens=True)`,
            where the return_hiddens argument is optional. The return value should be a tuple of the form (output, hiddens), where
            output is the output of the model and hiddens is a list of hidden states from the model. Only hidden states at certain
            indices will be used in the UNETR model.
            embedding_size: the size of the output of the image_encoder model.
            feature_size: the size of the features in the UNETR model.
            out_channels: the number of output channels.
            norm_name: the name of the normalization layer to use in the UNETR model.
            input_size: the size of the input image.
            output_size: the size of the output heatmap.
        """

        super().__init__()

        self.image_encoder = image_encoder

        embedding_size = embedding_size
        feature_size = feature_size  # divides embedding size

        # if the input size is greater than the output size, we need to downsample. 
        # however, we don't sample the input but rather the transformer intermediate outputs
        if input_size > output_size:
            self.downsample = torch.nn.MaxPool2d(input_size // output_size)
        else:
            self.downsample = torch.nn.Identity()

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=3,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            upsample_kernel_size=2,
            stride=1,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            num_layer=0, 
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            conv_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=embedding_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out_block = UnetOutBlock(
            spatial_dims=2,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def vit_out_to_conv_in(self, x):
        return x.permute(0, 3, 1, 2)

    def forward(self, x):
        _, hiddens = self.image_encoder(x, return_hiddens=True)
        x1 = self.downsample(x)
        x2 = self.downsample(self.vit_out_to_conv_in(hiddens[3]))
        x3 = self.downsample(self.vit_out_to_conv_in(hiddens[6]))
        x4 = self.downsample(self.vit_out_to_conv_in(hiddens[9]))
        x5 = self.downsample(self.vit_out_to_conv_in(hiddens[11]))

        enc1 = self.encoder1(x1)
        enc2 = self.encoder2(x2)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)
        dec4 = x5

        dec3 = self.decoder1(dec4, enc4)
        dec2 = self.decoder2(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder4(dec1, enc1)

        return self.out_block(dec0)
    