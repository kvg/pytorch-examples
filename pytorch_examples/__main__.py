import logging

import click
import click_log

import sys
import pkgutil
from datetime import datetime

import pytorch_examples
from . import __version__


logger = logging.getLogger("version")
click_log.basic_config(logger)
logger.handlers[0].formatter = logging.Formatter(
    "[%(levelname)s %(asctime)s %(name)8s] %(message)s", "%Y-%m-%d %H:%M:%S"
)


@click.group()
def cli():
    logger.info("Invoked via: pytorch-examples %s", " ".join(sys.argv))


@cli.command()
@click_log.simple_verbosity_option(logger)
def version():
    """Print the version of pytorch-examples."""
    click.echo(__version__)


# Dynamically find and import sub-commands (allows for plugins at run-time):
# An alternative would be to iterate through the following:
# pkgutil.iter_modules([os.path.dirname((inspect.getmodule(sys.modules[__name__]).__file__))])
for p in [p for p in pkgutil.iter_modules(pytorch_examples.__path__) if p.ispkg]:
    exec(f"from .{p.name} import command as {p.name}")
    exec(f"cli.add_command({p.name}.main)")


if __name__ == "__main__":
    cli()
