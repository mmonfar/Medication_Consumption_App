"""Medication consumption forecast dashboard."""

__all__ = ["create_app"]


def create_app():  # lazy import: keeps `medication_app.data` importable alone
    from .app import create_app as _create_app

    return _create_app()
