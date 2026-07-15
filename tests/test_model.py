"""Tests for core/utils/model.py (EmbeddingClassifier)."""
import torch

from core.utils.model import EmbeddingClassifier


def test_default_hidden_dim_matches_input_dim():
    model = EmbeddingClassifier(input_dim=32, hidden_dim=None, num_classes=5)
    assert model.hidden_dim == 32


def test_forward_output_shape():
    model = EmbeddingClassifier(input_dim=16, hidden_dim=8, num_classes=3)
    x = torch.randn(10, 16)
    logits = model(x)
    assert logits.shape == (10, 3)


def test_forward_return_embedding():
    model = EmbeddingClassifier(input_dim=16, hidden_dim=8, num_classes=3)
    x = torch.randn(5, 16)
    logits, embedding = model(x, return_embedding=True)
    assert logits.shape == (5, 3)
    assert embedding.shape == (5, 8)


def test_get_embedding_matches_forward_embedding_in_eval_mode():
    model = EmbeddingClassifier(input_dim=16, hidden_dim=8, num_classes=3, dropout_rate=0.0)
    model.eval()
    x = torch.randn(4, 16)
    _, embedding_from_forward = model(x, return_embedding=True)
    embedding_from_helper = model.get_embedding(x)
    assert torch.allclose(embedding_from_forward, embedding_from_helper)


def test_get_embedding_does_not_track_gradients():
    model = EmbeddingClassifier(input_dim=16, hidden_dim=8, num_classes=3)
    x = torch.randn(4, 16, requires_grad=True)
    embedding = model.get_embedding(x)
    assert embedding.requires_grad is False


def test_weight_initialization_is_nonzero_and_biases_are_zero():
    model = EmbeddingClassifier(input_dim=16, hidden_dim=8, num_classes=3)
    assert torch.any(model.projection.weight != 0)
    assert torch.any(model.classifier.weight != 0)
    assert torch.all(model.projection.bias == 0)
    assert torch.all(model.classifier.bias == 0)


def test_dropout_introduces_stochasticity_in_train_mode():
    torch.manual_seed(0)
    model = EmbeddingClassifier(input_dim=16, hidden_dim=64, num_classes=3, dropout_rate=0.9)
    model.train()
    x = torch.randn(1, 16)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)


def test_dropout_disabled_in_eval_mode_is_deterministic():
    model = EmbeddingClassifier(input_dim=16, hidden_dim=64, num_classes=3, dropout_rate=0.9)
    model.eval()
    x = torch.randn(1, 16)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)
