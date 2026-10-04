"""Vendored DL-GF beam-alignment code.

The learned probing / beam-synthesis architecture and the codebook helpers come
from the reference implementation of

    Y. Heng, J. Mo and J. G. Andrews, "Learning Site-Specific Probing Beams for
    Fast mmWave Beam Alignment", IEEE Trans. Wireless Commun., 2022,
    https://github.com/YuqiangHeng/DLGF

The files are kept close to the originals (imports adapted to the package
layout, plus a beamforming-gain loss and an evaluation helper added for this
study) so that they can be diffed against upstream. They are GPL-3.0 licensed;
see ``LICENSES/DLGF-GPL-3.0.txt`` and ``THIRD_PARTY_NOTICES.md``.
"""

from .codebooks import (UPA_DFT_codebook, ULA_DFT_codebook, dB_2_pow,
                        pow_2_dB, unravel_index)
from .models import (BF_loss, Joint_BF_Autoencoder,
                     Joint_Tx_Rx_Analog_Beamformer,
                     Joint_Tx_Rx_Analog_Beamformer_DFT, eval_model)

__all__ = [
    "BF_loss",
    "Joint_BF_Autoencoder",
    "Joint_Tx_Rx_Analog_Beamformer",
    "Joint_Tx_Rx_Analog_Beamformer_DFT",
    "eval_model",
    "UPA_DFT_codebook",
    "ULA_DFT_codebook",
    "dB_2_pow",
    "pow_2_dB",
    "unravel_index",
]
