from rich.logging import RichHandler
import logging
from rich.console import Console
import sys

# Create console with larger width for log messages and word wrap enabled
console = Console(width=150, stderr=True, soft_wrap=True)

# Create and configure the Rich handler
rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    markup=True,
    show_time=True,
    show_path=False,
    enable_link_path=False,
    log_time_format="[%X]",
    show_level=True,
    omit_repeated_times=False
)

# Get the root logger and remove any existing handlers
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    root_logger.removeHandler(handler)

# Configure the root logger
root_logger.addHandler(rich_handler)
root_logger.setLevel(logging.WARNING)

# Configure the format for the Rich handler
rich_handler.setFormatter(logging.Formatter("%(message)s"))

# Prevent duplicate logging
logging.getLogger("bespokelabs.curator").propagate = False

from .dataset import Dataset
from .llm.llm import LLM