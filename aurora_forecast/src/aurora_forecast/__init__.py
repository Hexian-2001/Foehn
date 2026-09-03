"""Aurora (Microsoft) real-time forecast — model-specific package.

Decoupled sibling of ``weathernext_forecast``. Owns only Aurora's variable
naming, units, grid and file layout. The model-agnostic download stage lives in
``opendata_download`` (reused unchanged); the GPU inference lives here.
"""
