import pytest
import torch

from kanfis.networks import IT1KANFISNetwork, IT2KANFISNetwork, build_network


@pytest.mark.parametrize("membership", ["gaussian", "sigmoid", "triangle"])
def test_it1_network_has_finite_outputs_and_gradients(membership):
    network = IT1KANFISNetwork(
        in_features=4,
        n_rules=3,
        out_features=2,
        n_layers=1,
        n_memberships=2,
        membership=membership,
    )
    x = torch.randn(5, 4, requires_grad=True)

    prediction, firing = network(x)
    prediction.sum().backward()

    assert prediction.shape == (5, 2)
    assert firing.shape == (5, 3)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(x.grad).all()


def test_it2_upper_widths_are_strictly_larger_than_lower_widths():
    network = IT2KANFISNetwork(4, 3, 1, n_memberships=2)

    lower, upper = network.layers[0].widths()

    assert lower.shape == (4, 3, 2)
    assert torch.all(upper > lower)


def test_build_network_rejects_non_gaussian_it2_membership():
    with pytest.raises(ValueError, match="Gaussian"):
        build_network(
            variant="it2",
            in_features=4,
            n_rules=3,
            out_features=1,
            membership="triangle",
        )


@pytest.mark.parametrize("variant", ["it1", "it2"])
def test_network_supports_one_rule(variant):
    network = build_network(variant, in_features=2, n_rules=1, out_features=1)

    prediction, firing = network(torch.randn(4, 2))

    assert prediction.shape == (4, 1)
    assert firing.shape == (4, 1)


def test_network_rejects_wrong_feature_count():
    network = IT1KANFISNetwork(4, 3, 1)

    with pytest.raises(ValueError, match="4 features"):
        network(torch.randn(2, 3))
