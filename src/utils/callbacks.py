from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch
from pytorch_lightning.callbacks import Callback, RichModelSummary, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.tree import Tree


THEMES: Dict[str, Dict[str, str]] = {
    "default": {
        "title": "bold #fb8500",
        "header": "bold #fb8500",
        "text": "#2ec4b6",
        "label": "#2ec4b6",
        "value": "bold #2ec4b6",
        "border": "#fb8500",
        "progress_bar": "green1",
        "progress_bar_pulse": "green1",
        "progress_bar_finished": "green1",
        "batch_progress": "green_yellow",
        "processing_speed": "#fb8500",
    },
    "cool_modern": {
        "title": "bold #4A90E2",
        "header": "bold #50E3C2",
        "text": "#D1E8E2",
        "label": "#50E3C2",
        "value": "#B8D8D8",
        "border": "#7B8D8E",
        "progress_bar": "#4A90E2",
        "progress_bar_pulse": "#4A90E2",
        "progress_bar_finished": "#417B82",
        "batch_progress": "#50E3C2",
    },
    "vibrant_high_contrast": {
        "title": "bold #FF6347",
        "header": "bold #FFD700",
        "text": "#FFFFFF",
        "label": "#FFD700",
        "value": "#1E90FF",
        "border": "#FF4500",
        "progress_bar": "#FF6347",
        "progress_bar_pulse": "#FF6347",
        "progress_bar_finished": "#FF4500",
        "batch_progress": "#FFD700",
    },
    "magenta": {
        "title": "bold #FF69B4",
        "header": "bold #FF69B4",
        "text": "#FFFFFF",
        "label": "#FF69B4",
        "value": "grey70",
        "border": "#8B008B",
        "progress_bar": "#FF1493",
        "progress_bar_pulse": "#FF69B4",
        "progress_bar_finished": "#C71585",
        "batch_progress": "#C71585",
    },
    "green_burgundy": {
        "title": "bold #556B2F",
        "header": "bold #6B8E23",
        "text": "#FFFFFF",
        "label": "#6B8E23",
        "value": "grey70",
        "border": "#8B0000",
        "progress_bar": "#556B2F",
        "progress_bar_pulse": "#6B8E23",
        "progress_bar_finished": "#8B0000",
        "batch_progress": "#8B0000",
    },
}


def _make_console(theme_name: str | None) -> Console:
    theme_cfg = THEMES.get(theme_name or "default", THEMES["default"])
    return Console(theme=Theme(theme_cfg))


class DatamoduleSummary(Callback):
    """通用数据模块摘要，打印训练与验证集的基本信息。"""

    def __init__(self, theme_name: str | None = "default", enabled: bool = True):
        super().__init__()
        self.console = _make_console(theme_name)
        self.enabled = enabled
        self._printed = False

    def on_fit_start(self, trainer, pl_module):  # type: ignore[override]
        if not self.enabled or not getattr(trainer, "is_global_zero", True) or self._printed:
            return
        if getattr(pl_module, "verbose", True):
            self.display_data_stats(trainer.datamodule)
            self._printed = True

    # -------------------------
    # helpers
    # -------------------------
    def _create_table(self) -> Table:
        return Table(box=None, show_header=False, min_width=32)

    def _add_rows(self, panel_title: str, table: Table, data: List[Tuple[str, str]]) -> None:
        if not data:
            return
        for left, right in data:
            table.add_row(f"[label]{left}[/label]", f"[value]{right}[/value]")
        panel = Panel(table, title=f"[title]{panel_title}[/title]", border_style="border", padding=(1, 1), expand=False)
        self.console.print(panel)

    def _add_tree(self, panel_title: str, tree_data: Dict[str, List[str]]) -> None:
        if not tree_data:
            return
        tree = Tree(panel_title, hide_root=True, guide_style="border")
        for node, children in tree_data.items():
            branch = tree.add(f"[label]{node}[/label]")
            for child in children:
                branch.add(f"[value]{child}[/value]")
        panel = Panel(tree, title=f"[title]{panel_title}[/title]", border_style="border", padding=(1, 2), expand=False)
        self.console.print(panel)

    def _push(self, rows: List[Tuple[str, str]], label: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            value = len(value)
        rows.append((label, str(value)))

    # -------------------------
    # public
    # -------------------------
    def display_data_stats(self, datamodule):
        self.console.print("\n")
        self._print_train_dataset(datamodule)
        self._print_val_datasets(datamodule)
        self._print_data_config(datamodule)
        self.console.print("\n")

    def _print_train_dataset(self, datamodule):
        train_ds = getattr(datamodule, "train_dataset", None)
        if train_ds is None:
            return

        rows: List[Tuple[str, str]] = []
        self._push(rows, "dataset", getattr(train_ds, "dataset_name", train_ds.__class__.__name__))
        self._push(rows, "length", len(train_ds))
        self._push(rows, "images", getattr(train_ds, "num_images", None))
        self._push(rows, "scenes", getattr(train_ds, "num_scenes", None))
        self._push(rows, "classes", getattr(train_ds, "num_classes", None))
        self._push(rows, "cities", getattr(train_ds, "cities", None))
        self._add_rows("Training dataset", self._create_table(), rows)

    def _print_val_datasets(self, datamodule):
        val_sets = getattr(datamodule, "val_datasets", None) or []
        if not val_sets:
            return

        tree_data: Dict[str, List[str]] = {}
        for ds in val_sets:
            name = getattr(ds, "dataset_name", ds.__class__.__name__)
            entries: List[str] = []
            queries = getattr(ds, "num_queries", None)
            references = getattr(ds, "num_references", None)
            if queries is not None:
                entries.append(f"queries {queries}")
            if references is not None:
                entries.append(f"references {references}")
            entries.append(f"length {len(ds)}")
            tree_data[name] = entries
        self._add_tree("Validation datasets", tree_data)

    def _print_data_config(self, datamodule):
        rows: List[Tuple[str, str]] = []
        self._push(rows, "train batch size", getattr(datamodule, "batch_size", None))
        self._push(rows, "num workers", getattr(datamodule, "num_workers", None))
        def _format_size(value: Any) -> str | Any:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return "x".join(str(int(v)) for v in value)
            return value

        self._push(rows, "train image size", _format_size(getattr(datamodule, "train_image_size", None)))
        self._push(rows, "val image size", _format_size(getattr(datamodule, "val_image_size", None)))

        extra_cfg = getattr(datamodule, "extra_data_config", {}) or {}
        for key, value in extra_cfg.items():
            # Convert extra config values to strings so lists/tuples/sets
            # are displayed as readable content instead of lengths.
            if isinstance(value, (list, tuple, set)):
                try:
                    s = ", ".join(str(x) for x in value)
                except Exception:
                    s = str(value)
            else:
                s = str(value)
            self._push(rows, key, s)

        self._add_rows("Data configuration", self._create_table(), rows)



class ModelFrameworkSummary(Callback):
    """Print a two-panel view for the GeoEncoder and framework settings."""

    def __init__(self, theme_name: str | None = "default", enabled: bool = True):
        super().__init__()
        self.console = _make_console(theme_name)
        self.enabled = enabled
        self._printed = False

    def _build_model_table(self, model: Any) -> Table:
        # match DatamoduleSummary table style
        table = Table(box=None, show_header=False, min_width=32)
        table.add_column("Field", justify="right")
        table.add_column("Value", justify="left")
        # Basic identity
        table.add_row(f"[label]Class[/label]", f"[value]{model.__class__.__name__}[/value]")
        # Backbone / aggregator identifiers
        table.add_row(f"[label]Backbone name[/label]", f"[value]{getattr(model, 'backbone_name', '-') }[/value]")
        backbone_args = getattr(model, 'backbone_args', {}) or {}
        if backbone_args:
            for k, v in backbone_args.items():
                table.add_row(f"[label]Backbone arg:{k}[/label]", f"[value]{str(v)}[/value]")
        table.add_row(f"[label]Aggregator name[/label]", f"[value]{getattr(model, 'aggregator_name', '-') }[/value]")
        aggregator_args = getattr(model, 'aggregator_args', {}) or {}
        if aggregator_args:
            for k, v in aggregator_args.items():
                table.add_row(f"[label]Aggregator arg:{k}[/label]", f"[value]{str(v)}[/value]")
        # channel dims
        table.add_row(f"[label]Backbone out dim[/label]", f"[value]{getattr(model, 'backbone_out_channels', '-') }[/value]")
        table.add_row(f"[label]Out dim[/label]", f"[value]{str(getattr(model, 'out_channels', '-'))}[/value]")

        # whitening / norm flags
        table.add_row(f"[label]Whitening[/label]", f"[value]{str(getattr(model, 'whitening', False))}[/value]")
        table.add_row(f"[label]Finetune whiten[/label]", f"[value]{str(getattr(model, 'finetune_whiten', False))}[/value]")
        table.add_row(f"[label]Final norm[/label]", f"[value]{str(getattr(model, 'final_norm', False))}[/value]")

        # params
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        table.add_row(f"[label]Params[/label]", f"[value]{params:,}[/value]")
        table.add_row(f"[label]Trainable[/label]", f"[value]{trainable:,}[/value]")

        return table

    def _build_framework_table(self, module: Any) -> Table:
        table = Table(box=None, show_header=False, min_width=32)
        table.add_column("Field", justify="right")
        table.add_column("Value", justify="left")

        table.add_row(f"[label]Lightning[/label]", f"[value]{module.__class__.__name__}[/value]")
        table.add_row(f"[label]use_proj[/label]", f"[value]{str(getattr(module, 'use_proj', False))}[/value]")
        table.add_row(f"[label]use_bn[/label]", f"[value]{str(getattr(module, 'use_bn', False))}[/value]")
        optimizer_cfg = getattr(module, "optimizer_cfg", {}) or {}
        optimizer_cfg = getattr(module, "optimizer_cfg", {}) or {}
        if optimizer_cfg:
            for k, v in optimizer_cfg.items():
                table.add_row(f"[label]Opt:{k}[/label]", f"[value]{str(v)}[/value]")

        scheduler_cfg = getattr(module, "scheduler_cfg", {}) or {}
        if scheduler_cfg:
            for k, v in scheduler_cfg.items():
                table.add_row(f"[label]Sch:{k}[/label]", f"[value]{str(v)}[/value]")
        return table

    def on_fit_start(self, trainer, pl_module):  # type: ignore[override]
        if not self.enabled or not getattr(trainer, "is_global_zero", True) or self._printed:
            return
        model = getattr(pl_module, "model", None)
        if model is None:
            return
        self._printed = True
        self.console.print(Panel(self._build_model_table(model), title=f"[title]GeoEncoder[/title]", border_style="border", padding=(1, 1), expand=False))
        self.console.print(Panel(self._build_framework_table(pl_module), title=f"[title]Framework[/title]", border_style="border", padding=(1, 1), expand=False))


class CustomRichProgressBar(RichProgressBar):
    def __init__(self, theme_name: str | None = "default"):
        if theme_name and theme_name in THEMES:
            super().__init__(
                leave=False,
                theme=RichProgressBarTheme(
                    description=THEMES[theme_name]["title"],
                    progress_bar=THEMES[theme_name]["progress_bar"],
                    progress_bar_finished=THEMES[theme_name]["progress_bar_finished"],
                    progress_bar_pulse=THEMES[theme_name]["progress_bar"],
                    batch_progress=THEMES[theme_name]["batch_progress"],
                    time=THEMES[theme_name]["text"],
                    processing_speed=THEMES[theme_name]["text"],
                    metrics=THEMES[theme_name]["label"],
                    metrics_text_delimiter="\n",
                    metrics_format=".6f",
                ),
            )
        else:
            super().__init__(leave=False)


class CustomRichModelSummary(RichModelSummary):
    def __init__(self, theme_name: str | None = "default"):
        header_style = THEMES.get(theme_name or "default", THEMES["default"]).get("header", None)
        if header_style:
            super().__init__(max_depth=1, header_style=header_style)
        else:
            super().__init__(max_depth=1)
