"""Downstream tasks that consume the generated channels."""

from . import beam_alignment, crnet, csi_compression

__all__ = ["beam_alignment", "crnet", "csi_compression"]
