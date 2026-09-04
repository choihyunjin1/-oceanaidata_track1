# P1 v50 local external-driver inventory

Audience: P1 experiment review. Date: 2026-09-01 KST. This was a read-only inventory of local manifests and already materialized, competition-allowed external/shared drivers. No network request, download, raw official P1 test/sample/submission/hidden read, or new data materialization occurred.

## Decision

`EXTERNAL_DRIVER_UNAVAILABLE`. The official FAQ permits public external data with attribution, but none of the already local driver families satisfies concurrent 2024–2025 coverage at two or more stations plus non-reconstructibility.

- Multi-station ERA5 covers G/I/S hourly from 2014 through 2023-12-31 only. It has no overlap with the 2024–2025 P1 train period.
- KMA ocean-buoy wind/weather/wave data has predeclared G/I/S proxies, but also ends on 2023-12-31; concurrent train overlap is zero.
- NASA POWER provides complete 2024 and 2025 meteorology only for S-ORS.
- The locally retrieved 2024–2025 ERA5 surface-forcing blocks are also S-ORS-only and cover three disjoint P2 windows rather than the full P1 train surface.
- Historical external CTD profiles are I-ORS-only, end in 2023, and their external-profile/density transfer recipe is already closed in the P1 registry.
- Tide phase uses no external observations and is deterministically reconstructible from timestamps; phase/window consistency is already a registered P1 family.

The strongest physically distinct drivers—actual wind, stress, heat flux, precipitation, pressure, and wave state—are therefore locally available either at multiple stations before the relevant period or concurrently at only S-ORS. Using a historical climatology as if it were the actual 2024–2025 causal driver would fabricate alignment and cannot resolve the repeated S-only action collapse.

A prospective contract was sealed before the decision: backward-asof alignment with three-hour staleness, current/6h-change/24h-mean/24h-std, three fixed seeds and fits, unchanged thresholds, v28/v33, and add-only removal=0. No family reached READY, so two READY preflights and exactly-once execution were inapplicable.

Counters: network fetches=0, downloads=0, runner=0, preflights=0, locks=0, fits=0, optimizer steps=0, target reads=0, actions=0, removals=0. Official, test, sample-submission, submission, hidden, CSV, and upload accesses were all zero.
