"""Project-scoped, timestamped run logging (W10).

Every entry point still uses plain print() throughout -- migrating hundreds of
call sites to logging.info() would be a large, risky refactor for no real
benefit here. Instead this puts a stdlib logging.FileHandler behind a thin
sys.stdout redirect: existing print() calls keep working completely
unchanged, but the file they land in is opened and managed by `logging`
rather than a hand-rolled `open(path, 'w')` (which truncated on every run and
wrote to a single fixed repo-root path shared across every project).
"""
import logging
import os
import sys
from datetime import datetime
from typing import Callable, Tuple


class _TeeToLogger:
    """Redirect sys.stdout writes to both the real console and a logging.Logger."""

    def __init__(self, logger: logging.Logger):
        self.terminal = sys.stdout
        self.logger = logger

    def write(self, message: str) -> None:
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, 'encoding', None) or 'utf-8'
            self.terminal.write(message.encode(encoding, errors='replace').decode(encoding))
        stripped = message.rstrip('\n')
        if stripped:
            self.logger.info(stripped)

    def flush(self) -> None:
        self.terminal.flush()


def setup_run_logging(log_dir: str, run_name: str) -> Tuple[Callable[[], None], str]:
    """Configure a timestamped, project-scoped log file and redirect sys.stdout to it.

    Returns (restore_fn, log_path). Call restore_fn() when the run is done to
    put sys.stdout back and close the file handler.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{run_name}_{timestamp}.log')

    logger = logging.getLogger(f'sisa.{run_name}.{timestamp}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    old_stdout = sys.stdout
    sys.stdout = _TeeToLogger(logger)

    def restore() -> None:
        sys.stdout = old_stdout
        handler.close()
        logger.removeHandler(handler)

    return restore, log_path
