from __future__ import annotations

from unittest.mock import patch

from pixsage.device import select_device


def test_select_device_prefers_cuda():
    with patch("pixsage.device._cuda_available", return_value=True), \
         patch("pixsage.device._mps_available", return_value=True):
        assert select_device() == "cuda"


def test_select_device_falls_back_to_mps():
    with patch("pixsage.device._cuda_available", return_value=False), \
         patch("pixsage.device._mps_available", return_value=True):
        assert select_device() == "mps"


def test_select_device_falls_back_to_cpu():
    with patch("pixsage.device._cuda_available", return_value=False), \
         patch("pixsage.device._mps_available", return_value=False):
        assert select_device() == "cpu"
