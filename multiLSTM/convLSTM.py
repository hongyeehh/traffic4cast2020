import torch.nn as nn
import numpy as np
import torch

from collections import OrderedDict


class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """

    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features, 2 * self.num_features, self.filter_size, 1, self.padding
            ),
            nn.GroupNorm(2 * self.num_features // 16, 2 * self.num_features),
            # nn.BatchNorm2d(2 * self.num_features),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features, self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 16, self.num_features),
            # nn.BatchNorm2d(self.num_features),
        )

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(0), self.num_features, self.shape[0], self.shape[1]).to(inputs.device)
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels, self.shape[0], self.shape[1]).to(htprev.device)
            else:
                x = inputs[:, index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev), 1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext

        return torch.stack(output_inner, dim=1), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """

    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features, 4 * self.num_features, self.filter_size, 1, self.padding
            ),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features),
        )

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(0), self.num_features, self.shape[0], self.shape[1]).to(inputs.device)
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).to(inputs.device)
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0], self.shape[1]).to(hx.device)
            else:
                x = inputs[:, index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner, axis=1), (hy, cy)


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if "deconv" in layer_name:
            upsample = nn.Upsample(mode="nearest", scale_factor=2)
            layers.append(("upsample_" + layer_name, upsample))
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append(("conv_" + layer_name, conv2d))
            if "relu" in layer_name:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
            elif "leaky" in layer_name:
                layers.append(("leaky_" + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif "conv" in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if "relu" in layer_name:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
            elif "leaky" in layer_name:
                layers.append(("leaky_" + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        elif "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, "stage" + str(index), make_layers(params))
            setattr(self, "rnn" + str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        b, t, c, h, w = input.size()
        input = torch.reshape(input, (-1, c, h, w))
        input = subnet(input)
        input = torch.reshape(input, (b, t, input.size(1), input.size(2), input.size(3)))
        outputs_stage, state_stage = rnn(input, None)

        return outputs_stage, state_stage

    # input: 5D B*T*C*H*W
    def forward(self, input):
        hidden_states = []

        for i in range(1, self.blocks + 1):
            input, state_stage = self.forward_by_stage(
                input, getattr(self, "stage" + str(i)), getattr(self, "rnn" + str(i))
            )
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, "rnn" + str(self.blocks - index), rnn)
            setattr(self, "stage" + str(self.blocks - index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, _ = rnn(input, state, seq_len=6)
        b, t, c, h, w = input.size()
        input = torch.reshape(input, (-1, c, h, w))
        input = subnet(input)
        input = torch.reshape(input, (b, t, input.size(1), input.size(2), input.size(3)))

        return input

    def forward(self, hidden_states):

        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, "stage4"), getattr(self, "rnn4"))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(
                input, hidden_states[i - 1], getattr(self, "stage" + str(i)), getattr(self, "rnn" + str(i))
            )
        return input


class EF(nn.Module):
    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output


# convlstm_encoder_params = [
#     [
#         OrderedDict({"conv1_leaky_1": [8, 32, 3, 1, 1], "pool_2": [2, 2, 1]}),
#         OrderedDict({"conv2_leaky_1": [64, 192, 3, 1, 1], "pool_2": [2, 2, 1]}),
#         OrderedDict({"conv3_leaky_1": [192, 192, 3, 1, 1], "pool_2": [2, 2, 1]}),
#     ],
#     [
#         ConvLSTM(
#             input_channel=32, num_filter=64, b_h_w=(config["batch_size"], 64, 64), kernel_size=3, stride=1, padding=1
#         ),
#         ConvLSTM(
#             input_channel=192, num_filter=192, b_h_w=(config["batch_size"], 32, 32), kernel_size=3, stride=1, padding=1
#         ),
#         ConvLSTM(
#             input_channel=192, num_filter=192, b_h_w=(config["batch_size"], 16, 16), kernel_size=3, stride=1, padding=1
#         ),
#     ],
# ]

# convlstm_forecaster_params = [
#     [
#         OrderedDict({"deconv1_leaky_1": [192, 192, 3, 1, 1]}),
#         OrderedDict({"deconv2_leaky_1": [192, 64, 3, 1, 1]}),
#         OrderedDict(
#             {"deconv3_leaky_1": [64, 32, 3, 1, 1], "conv3_leaky_2": [32, 16, 3, 1, 1], "conv3_3": [16, 8, 1, 1, 0]}
#         ),
#     ],
#     [
#         ConvLSTM(
#             input_channel=192, num_filter=192, b_h_w=(config["batch_size"], 16, 16), kernel_size=3, stride=1, padding=1
#         ),
#         ConvLSTM(
#             input_channel=192, num_filter=192, b_h_w=(config["batch_size"], 32, 32), kernel_size=3, stride=1, padding=1
#         ),
#         ConvLSTM(
#             input_channel=64, num_filter=64, b_h_w=(config["batch_size"], 64, 64), kernel_size=3, stride=1, padding=1
#         ),
#     ],
# ]

if __name__ == "__main__":
    from config import config

    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(config["device"])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(config["device"])
    encoder_forecaster = EF(encoder, forecaster).to(config["device"])

    input_x = torch.rand((4, 12, 8, 128, 128))
    out = encoder_forecaster(input_x)

    print(out.shape)
