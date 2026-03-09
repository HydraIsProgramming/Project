"""Compatibility entrypoint for 2D interactive app."""

from rl_armMotion.two_d.gui.app import *  # noqa: F401,F403
from rl_armMotion.two_d.gui.app import main


if __name__ == "__main__":
    main()
