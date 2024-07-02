import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             'src')))
from PerceptualLoss import PerceptualLoss, VGG16FeatureExtractor  # noqa: E402
import torch  # noqa: E402
from torch.nn import MSELoss  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def setup_perceptual_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = VGG16FeatureExtractor().to(device)
    perceptual_loss = PerceptualLoss(feature_extractor, MSELoss())
    return perceptual_loss, device


def test_perceptual_loss_equality(setup_perceptual_loss):
    perceptual_loss, device = setup_perceptual_loss

    input_tensor = torch.rand(1, 1, 224, 224, device=device)
    target_tensor = input_tensor.clone()

    loss = perceptual_loss.get_loss(input_tensor, target_tensor)
    assert_bool = loss.item() == pytest.approx(0, abs=1e-6)
    assert assert_bool, "Loss should be close to " + \
                        "0 for identical inputs and targets."


def test_perceptual_loss_difference(setup_perceptual_loss):
    perceptual_loss, device = setup_perceptual_loss

    input_tensor = torch.rand(1, 1, 224, 224, device=device)
    target_tensor = torch.zeros(1, 1, 224, 224, device=device)

    loss = perceptual_loss.get_loss(input_tensor, target_tensor)

    assert loss.item() > 0.1, 'Loss should be high for significantly ' + \
                              'different inputs and targets.'
