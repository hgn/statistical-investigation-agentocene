"""Analysis layer.

Reads only the JSONL records emitted by the gathering layer; never touches git
or the network. Modules:

* ``inclusion`` -- the pre-registered inclusion gate (concept.md sec. 5.1)
* ``manifest``  -- per-repo audit file (URL + used?/contributors/p_ai/...)
* ``panel``     -- consolidate records into tidy CSV/Parquet panels (TODO)
* ``stats``     -- excess-churn / dose-response / propensity (TODO)
* ``plots``     -- matplotlib figures (TODO)
"""
