"""Console identity for GeoMetricLab command-line runs."""

import os

from rich.console import Console


STARTUP_BANNER = r"""
   ____ __  __ _          _
  / ___|  \/  | |    __ _| |__
 | |  _| |\/| | |   / _` | '_ \
 | |_| | |  | | |__| (_| | |_) |
  \____|_|  |_|_____\__,_|_.__/

 Welcome to GeoMetricLab
 Geometric metric learning for visual retrieval
 Developed by Suxilan
""".strip("\n")


def show_startup_banner(console: Console | None = None) -> None:
    """Show the startup identity only on the primary process."""
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    if int(rank) == 0:
        from rich.panel import Panel

        output = console if console is not None else Console()
        output.print(
            Panel(STARTUP_BANNER, border_style="cyan", padding=(1, 2), expand=False),
            highlight=False,
        )
