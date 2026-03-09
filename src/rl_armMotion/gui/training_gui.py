"""Compatibility entrypoint for 2D training GUI."""

from rl_armMotion.two_d.gui.training_gui import *  # noqa: F401,F403


if __name__ == "__main__":
    from rl_armMotion.two_d.gui.training_gui import _parse_args, TrainingGUI

    args = _parse_args()
    gui = TrainingGUI(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        algorithm=args.algorithm,
    )
    gui.run()
