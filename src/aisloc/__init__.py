"""AI-induced SLOC velocity study.

Two cleanly separated halves:

* ``aisloc.gather`` and ``aisloc.mining`` / ``aisloc.sources`` / ``aisloc.resources``
  form the *data gathering* layer. It depends only on the Python standard
  library and the ``git`` binary, so it can run on constrained collection hosts
  and be pointed at a different forge (public GitHub now, on-prem GitLab later)
  by swapping a single ``RepoSource`` implementation.

* The *analysis* layer (panel build, statistics, plots, propensity model) lives
  behind the ``requirements.txt`` dependencies and only ever reads the JSONL
  records emitted by gathering. It never touches git or the network.
"""

__version__ = "0.1.0"
