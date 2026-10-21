# Wind & solar outlook for Poland

![tests](https://github.com/kwwaleszczynski-del/pl-res-forecast-climate/actions/workflows/ci.yml/badge.svg)

A **probabilistic forecast of national wind and solar generation in Poland for the next seven days**, built only
from open data: ensemble weather forecasts (ECMWF, DWD, NOAA via Open-Meteo) are turned into power with physical
models, calibrated every day against the generation reported by the transmission system operator PSE, and published
as a fan of P10/P50/P90 on a static website that refreshes itself with GitHub Actions. A rolling-origin
**hindcast** shows how good the method is, and an **ERA5 resource climatology** puts the current year in a 35-year
context (anomaly maps, "how was this year for renewables", P50/P90 of annual yield).

<!--RESULTS-->

## How it works

```
Open-Meteo ensembles ──► 48 grid points ──► pvlib / fleet power curve ──► national physical MW per member
(ECMWF 51, ICON 40,      (1° lattice,        (capacity-weighted by          │
 GEFS 31 members)         public capacity)    voivodeship)                   ▼
                                                              daily MOS per model and lead, refitted on the
PSE his-wlk-cal (actual wind & PV) ─────────────────────────► last observed days + out-of-sample errors
                                                                             │
                                                                             ▼
                                              dressed multi-model ensemble ─► P05..P95 fan, daily energy, risk
```

1. **Grid and capacity.** A 1° lattice clipped to Poland gives 48 points, each placed at the centroid of the
   Polish part of its cell. Each point represents a share of the public installed capacity of the voivodeships it
   overlaps: URE's register (11.4 GW wind, 12.8 GW licensed PV, June 2026) plus about 14 GW of rooftop PV
   distributed by population. No individual plant locations are used.
2. **Weather to power.** Solar: hourly GHI is split into beam and diffuse (Erbs), transposed onto a mix of
   south-facing and east/west planes (Hay-Davies), corrected for incidence angle and cell temperature (Faiman) and
   converted with a PVWatts-style model with inverter clipping (pvlib). Wind: 80/100 m wind is shifted to a 100 m hub
   height and passed through a generic turbine curve smoothed over the spread of speeds within a cell (a
   multi-turbine power curve).
3. **Calibration (MOS).** For every model and lead day, `y = x·(a + b·min(x, x_max) + c·f) + d` maps the physical
   output `x` onto PSE generation `y`; for solar, `f` are the sun's azimuth and a low-sun indicator, which let the
   fit correct the daily profile (row shading in winter, orientation mix). The fit is redone every day on the
   latest observed window, so the fleet's growth, curtailment and model biases are absorbed without a capacity time
   series.
4. **Uncertainty.** The errors of earlier out-of-sample forecasts, grouped by output level, form an empirical error
   distribution. Every calibrated ensemble member is dressed with the errors of its own model, scaled by
   `k = sqrt(max(k_min², 1 − s²/σ²))` (`s` = ensemble spread, `σ` = typical error): a single deterministic run gets
   the full error distribution, a widely spread ensemble only a fraction of it. Each model gets equal weight.
5. **Products.** Hourly quantiles P05...P95 for today to D+7, daily energy with ranges, the solar peak, and the
   probability that a day will be among the weakest 10 % of wind + solar days of the past year.

## Verification

<!--VERIFICATION-->

## Resource climate

<!--CLIMATE-->

## Running it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"    # Linux/macOS: .venv/bin/python
python -m res_forecast backfill        # one-off history download; resumable, paced to Open-Meteo's free limits
python -m res_forecast hindcast --tune # tune on the first year of the hindcast, verify on the rest
python -m res_forecast climate         # ERA5 climatology, writes the base tables to results/climate
python -m res_forecast update          # daily run: live forecast, archive, site/index.html
pytest -q                              # offline tests
```

The backfill needs about two days of the free Open-Meteo budget (roughly 20 000 weighted calls, mostly 35 years of
daily ERA5 on 48 points); a daily update costs about 1 100 calls. Downloads are cached as Parquet in `data/`
(git-ignored). `results/` holds the small committed outputs: hindcast scores and parameters, and the ERA5 base
tables that let the daily run fetch only the current year.

**Automation.** `.github/workflows/daily.yml` runs `update` and deploys `site/` to GitHub Pages. Issued forecasts
are appended to a separate `forecast-archive` branch, so the main branch does not grow; the archive feeds the live
verification on the page. The workflow's cron line is commented out until scheduling is switched on.

## Data and licences

| Data | Source | Licence / terms |
|---|---|---|
| Ensemble forecasts, archived runs, ERA5 | [Open-Meteo](https://open-meteo.com/) APIs (free, non-commercial tier) | CC BY 4.0 |
| ECMWF IFS / IFS ENS | European Centre for Medium-Range Weather Forecasts, via Open-Meteo | CC BY 4.0 |
| ICON / ICON-EPS | Deutscher Wetterdienst, via Open-Meteo | DWD open data, attribution |
| GFS / GEFS | NOAA NCEP, via Open-Meteo | US government public domain |
| ERA5 reanalysis | Copernicus Climate Change Service (C3S), via Open-Meteo | Contains modified Copernicus Climate Change Service information; neither the European Commission nor ECMWF is responsible for any use |
| Wind and solar generation | [PSE S.A.](https://raporty.pse.pl/) reports API (`his-wlk-cal`) | public reports, attribution |
| Installed capacity | [URE](https://www.ure.gov.pl/) "Mapa OZE", 30 Jun 2026 (voivodeship sums only) | public information |
| Population | GUS, 31 Dec 2025 | public statistics |
| Voivodeship borders | [Natural Earth](https://www.naturalearthdata.com/) 1:10m admin-1 | public domain |

This project is independent and not affiliated with or endorsed by any of these organisations.

## Limitations

- National totals only. The 1° grid and voivodeship-level capacity are too coarse for regional or plant-level use.
- Curtailment, negative-price shutdowns and grid constraints are only captured statistically by the rolling
  calibration; days with unusual curtailment are under-predicted by the error model.
- PSE's solar figure includes an estimate of rooftop generation and is revised; it is treated as the truth here.
- Archived ensemble members are not available, so the hindcast verifies a three-member multi-model ensemble of the
  deterministic runs, not the live 122-member ensemble. The live forecast is verified separately as its archive
  grows.
- The hindcast starts in mid-2024 (PSE data), ICON and GFS runs from March 2025, so rare extremes are
  under-sampled.
- The annual yields use generic equipment and today's fleet layout; they describe the resource, not the output of a
  particular plant.

## Project layout

```
src/res_forecast/
  openmeteo.py    API client with weighted rate limiting      pse.py         PSE generation
  geo.py          grid, voivodeships, capacity weights        history.py     cached archived runs and ERA5
  power.py        pvlib PV fleet, multi-turbine wind curve    physical.py    national physical output
  calibration.py  MOS, error model, dressing, CRPS            hindcast.py    rolling-origin hindcast
  forecast.py     live ensemble forecast and KPIs             verify.py      scores
  climate.py      ERA5 anomalies and annual yields            archive.py     archive and live verification
  site.py         static website (templates/index.html)       plots.py       README charts
tools/build_static_data.py   builds the capacity table and borders from URE and Natural Earth
```

## License

MIT, see [LICENSE](LICENSE). Data remain under the licences of their providers listed above.
