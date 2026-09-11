import unittest

import torch

from src.model.full_model import SleepStagingModel


def _config(causal=True):
    return {
        "model": {
            "architecture": "gru",
            "mrcnn_channels_1": 16,
            "mrcnn_channels_2": 16,
            "gru_hidden_size": 32,
            "gru_num_layers": 2,
            "gru_dropout": 0.1,
            "num_classes": 5,
            "causal": causal,
        }
    }


class GRUArchitectureTests(unittest.TestCase):
    def test_causal_gru_output_shape_and_parameter_budget(self):
        model = SleepStagingModel(config=_config()).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 12, 100))

        self.assertEqual(output.shape, (2, 12, 5))
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 50_000)

    def test_causal_gru_does_not_propagate_future_input_backward(self):
        torch.manual_seed(31)
        model = SleepStagingModel(config=_config()).eval()
        signal = torch.randn(1, 12, 100)
        changed = signal.clone()
        changed[:, 6:, :] += 100.0

        with torch.no_grad():
            original = model(signal)
            perturbed = model(changed)

        torch.testing.assert_close(original[:, :6], perturbed[:, :6], rtol=0, atol=0)

    def test_noncausal_gru_control_can_propagate_future_input(self):
        torch.manual_seed(31)
        model = SleepStagingModel(config=_config(causal=False)).eval()
        signal = torch.randn(1, 12, 100)
        changed = signal.clone()
        changed[:, 6:, :] += 100.0

        with torch.no_grad():
            original = model(signal)
            perturbed = model(changed)

        self.assertFalse(torch.equal(original[:, :6], perturbed[:, :6]))


if __name__ == "__main__":
    unittest.main()