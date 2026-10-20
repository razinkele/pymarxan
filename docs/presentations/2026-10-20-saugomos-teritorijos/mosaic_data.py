"""Slide data for the reed-bed mosaic slide, from ``examples/curonian_spit_reedbeds.py``.

Solves the shore without and with the default edge reward and writes ``mosaic.json`` next to
this file for ``build.js``. Run under the shiny env: ``python mosaic_data.py``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "examples"))
import curonian_spit_reedbeds as reed  # noqa: E402


def main() -> None:
    plain, mosaic = reed.reward_sweep([0.0, reed.DEFAULT_EDGE_REWARD])
    data = {
        "plain": asdict(plain),
        "mosaic": asdict(mosaic),
        "settlements": reed.SETTLEMENTS,
        "n_segments": reed.N_SEGMENTS,
    }
    (HERE / "mosaic.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    print(f"edges {plain.edges} -> {mosaic.edges}, "
          f"cost {plain.cutting_cost:.1f} -> {mosaic.cutting_cost:.1f} k€")


if __name__ == "__main__":
    main()
