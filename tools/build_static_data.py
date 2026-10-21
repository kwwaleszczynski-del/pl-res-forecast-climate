"""Build the small static inputs shipped with the package (run once, results are committed).

Outputs (in src/res_forecast/data/):
    voivodeships.geojson      16 Polish voivodeships, simplified, from Natural Earth 1:10m admin-1 (public domain)
    capacity_voivodeship.csv  installed wind and PV capacity per voivodeship (MW)

Capacity sources:
    - URE "Mapa OZE" (installations holding a licence or registered as small installations), state on 30 Jun 2026:
      https://www.ure.gov.pl/pl/oze/potencjal-krajowy-oze/8108 ; only voivodeship sums are kept.
    - Micro-installations (<= 50 kW, mostly rooftop PV) are not in the URE list. PSE reported ~27.1 GW of PV in May
      2026, the URE list covers ~12.8 GW, so ~14 GW of micro PV is distributed by population (GUS, 31 Dec 2025).

Usage:
    python tools/build_static_data.py --ne-geojson ne_10m_admin_1_states_provinces.geojson --ure-xlsx mapaozena30-06-2026.xlsx
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("build_static_data")

MICRO_PV_MW = 14_000.0
# GUS, population on 31 Dec 2025
POPULATION = {
    "dolnośląskie": 2_857_273, "kujawsko-pomorskie": 1_971_891, "lubelskie": 1_980_772, "lubuskie": 963_601,
    "łódzkie": 2_328_825, "małopolskie": 3_429_342, "mazowieckie": 5_505_857, "opolskie": 923_536,
    "podkarpackie": 2_053_760, "podlaskie": 1_126_679, "pomorskie": 2_358_409, "śląskie": 4_261_792,
    "świętokrzyskie": 1_147_293, "warmińsko-mazurskie": 1_339_498, "wielkopolskie": 3_471_302,
    "zachodniopomorskie": 1_612_680,
}


def _ring(coords: list, ndigits: int = 3) -> list:
    out = []
    for lon, lat in coords:
        p = [round(lon, ndigits), round(lat, ndigits)]
        if not out or p != out[-1]:
            out.append(p)
    if out[0] != out[-1]:
        out.append(out[0])
    return out


def build_geojson(ne_path: Path) -> dict:
    src = json.loads(ne_path.read_text(encoding="utf-8"))
    features = []
    for f in src["features"]:
        p = f["properties"]
        if p.get("adm0_a3") != "POL":
            continue
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        polys = [[_ring(r) for r in poly] for poly in polys]
        name = p["name_pl"].replace("województwo ", "").strip()
        features.append({
            "type": "Feature",
            "properties": {"code": p["iso_3166_2"], "name": name},
            "geometry": {"type": "MultiPolygon", "coordinates": polys},
        })
    features.sort(key=lambda f: f["properties"]["code"])
    if len(features) != 16:
        raise ValueError(f"expected 16 voivodeships, got {len(features)}")
    return {"type": "FeatureCollection", "features": features}


def build_capacity(ure_path: Path, geo: dict) -> pd.DataFrame:
    df = pd.read_excel(ure_path, sheet_name=0, usecols=range(8))
    df.columns = ["lp", "dkn", "loc", "voiv", "powiat", "gmina", "mw", "type"]
    df["voiv"] = df["voiv"].astype(str).str.strip().str.lower()
    agg = (df[df["type"].isin(["WIL", "PVA"])]
           .pivot_table(index="voiv", columns="type", values="mw", aggfunc="sum").fillna(0.0))
    codes = {f["properties"]["name"]: f["properties"]["code"] for f in geo["features"]}
    pop = pd.Series(POPULATION)
    out = pd.DataFrame({
        "code": [codes[n] for n in pop.index],
        "name": pop.index,
        "population": pop.values,
        "wind_mw_ure": agg.reindex(pop.index)["WIL"].values,
        "pv_mw_ure": agg.reindex(pop.index)["PVA"].values,
    })
    out["pv_mw_micro_est"] = MICRO_PV_MW * out["population"] / out["population"].sum()
    out["wind_mw"] = out["wind_mw_ure"]
    out["pv_mw"] = out["pv_mw_ure"] + out["pv_mw_micro_est"]
    return out.sort_values("code").round(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ne-geojson", type=Path, required=True, help="Natural Earth 10m admin-1 GeoJSON")
    ap.add_argument("--ure-xlsx", type=Path, required=True, help="URE 'Mapa OZE' workbook")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "src/res_forecast/data")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    geo = build_geojson(args.ne_geojson)
    (args.out_dir / "voivodeships.geojson").write_text(json.dumps(geo, ensure_ascii=False, separators=(",", ":")),
                                                       encoding="utf-8")
    cap = build_capacity(args.ure_xlsx, geo)
    cap.to_csv(args.out_dir / "capacity_voivodeship.csv", index=False)
    log.info("wind %.0f MW, PV %.0f MW (URE %.0f + micro %.0f)", cap["wind_mw"].sum(), cap["pv_mw"].sum(),
             cap["pv_mw_ure"].sum(), cap["pv_mw_micro_est"].sum())


if __name__ == "__main__":
    main()
