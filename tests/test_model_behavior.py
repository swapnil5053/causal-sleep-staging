import unittest

import torch

from src.model.attention import CausalSelfAttention
from src.model.causal_conv import CausalConv1d
from src.model.full_model import SleepStagingModel


class CausalConvTests(unittest.TestCase):
    def test_convolution_preserves_sequence_length(self):
        layer = CausalConv1d(2, 4, kernel_size=5, dilation=3)
        x = torch.randn(3, 2, 41)

        output = layer(x)

        self.assertEqual(output.shape, (3, 4, 41))

    def test_future_input_does_not_change_earlier_output(self):
        torch.manual_seed(7)
        layer = CausalConv1d(1, 3, kernel_size=5, dilation=2)
        x = torch.randn(1, 1, 24)
        changed = x.clone()
        changed[:, :, 12:] += 100.0

        original_output = layer(x)
        changed_output = layer(changed)

        torch.testing.assert_close(
            original_output[:, :, :12],
            changed_output[:, :, :12],
            rtol=0,
            atol=0,
        )


class AttentionTests(unittest.TestCase):
    def test_causal_attention_ignores_future_positions(self):
        torch.manual_seed(11)
        layer = CausalSelfAttention(embed_dim=8, num_heads=2, dropout=0.0, causal=True)
        layer.eval()
        x = torch.randn(2, 10, 8)
        changed = x.clone()
        changed[:, 6:, :] += 100.0

        with torch.no_grad():
            original_output = layer(x)
            changed_output = layer(changed)

        torch.testing.assert_close(
            original_output[:, :6, :],
            changed_output[:, :6, :],
            rtol=0,
            atol=0,
        )


class FullModelTests(unittest.TestCase):
    @staticmethod
    def _config():
        return {
            "model": {
                "mrcnn_channels_1": 16,
                "mrcnn_channels_2": 16,
                "tcn_channels": [32, 32, 32],
                "tcn_kernel_size": 3,
                "tcn_dilations": [1, 2, 4],
                "tcn_dropout": 0.2,
                "attn_num_heads": 4,
                "attn_dropout": 0.1,
                "num_classes": 5,
                "causal": True,
            }
        }

    def test_output_shape_and_parameter_count(self):
        model = SleepStagingModel(config=self._config())
        model.eval()
        x = torch.randn(2, 12, 100)

        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (2, 12, 5))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 30_757)

    def test_model_layers_do_not_propagate_future_input_backward(self):
        torch.manual_seed(19)
        model = SleepStagingModel(config=self._config())
        model.eval()
        x = torch.randn(1, 12, 100)
        changed = x.clone()
        changed[:, 6:, :] += 100.0

        with torch.no_grad():
            original_output = model(x)
            changed_output = model(changed)

        torch.testing.assert_close(
            original_output[:, :6, :],
            changed_output[:, :6, :],
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
