"""Run GeneICL's ``benchmark.py`` frozen-embedding probe under a key it does not list.

Executed by ``run_probe.py`` with the GeneICL eval environment's python, so it imports
nothing from this repository:

    <geneicl python> _probe_shim.py <geneicl repo> <probe key> key=value ...

``benchmark.main`` only accepts models in its ``EMB_MODELS`` set for the embedding
branch (the PCA-95% front-end and the untuned elastic-net / EN-logistic probe), so the
key is added to that set before ``main`` parses the Hydra overrides. Nothing else in
the harness changes: splits, probe, metrics and output files are GeneICL's own.
"""

import os
import sys


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: _probe_shim.py <geneicl repo> <probe key> [key=value ...]")
    repo, key, overrides = sys.argv[1], sys.argv[2], sys.argv[3:]
    bench_dir = os.path.join(repo, "eval", "benchmark")
    sys.path.insert(0, bench_dir)
    import benchmark as bm

    bm.EMB_MODELS.add(key)
    sys.argv = [bm.__file__, f"model={key}", *overrides]
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    # Imported, main's module is "benchmark", which Hydra would search as a config package;
    # as "__main__" it reads the in-code ConfigStore from the file, as when run as a script.
    os.environ["HYDRA_MAIN_MODULE"] = "__main__"
    bm.main()


if __name__ == "__main__":
    main()
