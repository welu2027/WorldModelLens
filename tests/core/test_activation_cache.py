"""Tests for ActivationCache."""

import pytest
import torch


def test_single_indexing(fake_cache):
    """Test single element indexing."""
    val = fake_cache["state", 0]
    assert isinstance(val, torch.Tensor)
    assert val.shape[0] == 32


def test_slice_indexing(fake_cache):
    """Test slice indexing."""
    vals = fake_cache["state", :]
    assert vals.shape[0] == 10
    assert vals.shape[1] == 32


def test_component_names(fake_cache):
    """Test component names."""
    names = fake_cache.component_names
    assert "state" in names
    assert "posterior" in names


def test_filter(fake_cache):
    """Test filtering by component."""
    filtered = fake_cache.filter("state")
    assert "state" in filtered.component_names
    assert len(filtered.component_names) > 0


def test_to_device(fake_cache):
    """Test moving to device."""
    device = torch.device("cpu")
    moved = fake_cache.to_device(device)
    val = moved["state", 0]
    assert val.device == device


def test_detach(fake_cache):
    """Test detaching."""
    fake_cache["state", 0].requires_grad = True
    detached = fake_cache.detach()
    assert not detached["state", 0].requires_grad
