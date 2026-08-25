from losses.clap import ClapLoss
from losses.fidelity import pingpong_fidelity_loss
from losses.hinge_clap import HingeClapLoss

__all__ = [
    "ClapLoss",
    "HingeClapLoss",
    "pingpong_fidelity_loss",
]
