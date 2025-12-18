"""
Simulation Driver Package
=========================

This package contains modules for 3D simulation engine and utilities.

Modules:
- engine: Main simulation engine components
- utils: Utility functions for projection and calculations
"""

# Import key functions for easier access
from .utils import project_points, find_closest_points

__all__ = ['project_points', 'find_closest_points']