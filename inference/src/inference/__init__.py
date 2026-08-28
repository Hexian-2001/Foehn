"""Model-agnostic forecast inference orchestration layer.

This package is deliberately decoupled from any specific model. It defines:

* ``inference.models.base.ModelRunner`` — the interface every model must
  implement (load weights, run, return an ``xarray.Dataset`` of predictions).
* ``inference.registry`` — a name → :class:`ModelSpec` table, so a model is
  addressed by a CLI string (``--model graphcast``) rather than by code.
* ``inference.inputs`` — locating the input .nc file and reading its IC time.
* ``inference.saver`` — writing results, independent of the model that made
  them.

Adding a future model (e.g. a 1-hour-step model) means adding a
``ModelRunner`` subclass and one registry entry — nothing else changes.
"""
