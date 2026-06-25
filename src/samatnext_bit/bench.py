from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .train import train_mode
from .utils import device, load_config, timestamp, versions, write_json


DEFAULT_MODES = [
    "fp_chainrule",
    "fp_mono_update_every_2",
    "bitnet_fake_ternary_chainrule",
    "bitnet_fake_ternary_mono_update_every_2",
    "bitnet_int8_ternary_legacy_chainrule",
    "bitnet_int8_ternary_legacy_mono_update_every_2",
    "bitnet_two_bit_ternary_legacy_chainrule",
    "bitnet_two_bit_ternary_legacy_mono_update_every_2",
    "bitnet_base3_packed_ternary_chainrule",
    "bitnet_base3_packed_ternary_mono_update_every_2",
    "bitnet_base3_lut_ternary_chainrule",
    "bitnet_base3_lut_ternary_mono_update_every_2",
    "bitnet_base3_tile_dot_ternary_chainrule",
    "bitnet_base3_tile_dot_ternary_mono_update_every_2",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--modes", default=",".join(DEFAULT_MODES))
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()
    cfg = load_config(args.config)
    dev = device()
    results = []
    print(f"device={dev} gpu={versions()['gpu']}", flush=True)
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"running {mode}", flush=True)
        try:
            r = train_mode(cfg, mode, dev)
            row = asdict(r)
        except Exception as exc:
            row = {"mode": mode, "completed": False, "error": repr(exc)}
        results.append(row)
        print(row, flush=True)
    payload = {"config": args.config, "versions": versions(), "results": results}
    out = Path(args.out_dir) / f"{Path(args.config).stem}_{timestamp()}" / "results.json"
    write_json(out, payload)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
