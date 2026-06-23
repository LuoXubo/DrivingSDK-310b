# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
from pathlib import Path

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

_UTILS = Path(__file__).resolve().parents[4] / 'tools' / 'loss_curve_utils.py'


def _load_save_loss_curves():
    spec = importlib.util.spec_from_file_location('loss_curve_utils', _UTILS)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.save_loss_curves


@HOOKS.register_module()
class LossCurveHook(Hook):
    """Save loss curves to work_dir after training finishes."""

    def __init__(self, out_prefix='loss_curves', save_plot=True):
        self.out_prefix = out_prefix
        self.save_plot = save_plot

    @master_only
    def after_run(self, runner):
        work_dir = Path(runner.work_dir)
        timestamp = getattr(runner, 'timestamp', None)
        try:
            save_loss_curves = _load_save_loss_curves()
            saved = save_loss_curves(
                work_dir=work_dir,
                timestamp=timestamp,
                out_prefix=self.out_prefix,
                save_plot=self.save_plot,
            )
            runner.logger.info('Saved training loss curves: %s', saved)
        except Exception as exc:
            runner.logger.warning('Failed to save loss curves: %s', exc)
