"""
ReadTime Wizard Backend Package
A simple API for estimating book reading times and generating reading schedules.
"""

__version__ = "1.0.0"
__author__ = "Ravi Botla"
__description__ = "A book reading time estimation service"

# You can optionally expose key classes/functions at the package level
from .readtime_model import ReadTimeModel
from .main import app

__all__ = ['ReadTimeModel', 'app']