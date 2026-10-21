# Power recording (Jetson tegrastats)

**`jetson_clocks`: OFF.** Keep `nvpmodel` fixed (25W).

## cpu-beam A/B (current branch: `cpu-beam` @ `664bb1f`)

Hybrid path is GPU scan + **CPU beam** (`--cpu-beam=yes`), not full CPU decode. Rebuild first:

```bash
# usual Jetson make (howto.md), then:
./docs-riley/power-sweep --cpu-beam
# smoke: ./docs-riley/power-sweep --cpu-beam --only fast_1k,hac_1k
```

| label | meaning | flags (plus `-C 128 -c 12288 -K 4096 -p 150`) |
|-------|---------|-----------------------------------------------|
| `cb_off` | GPU beam baseline | `--cpu-beam=no` |
| `cb_on` | CPU beam hybrid | `--cpu-beam=yes` |

Expect `cb_on` to be **much slower** (~2.5–3× from prior timing) and to burn more host CPU / energy. HAC 20k `cb_on` can take ~40+ minutes.

## Other modes (other commits)

```bash
./docs-riley/power-sweep --pre
./docs-riley/power-sweep --overlap
./docs-riley/power-sweep --loadimb
```

## PC charts

```bash
source .venv-power/bin/activate
python fetch-and-plot-power.py
```
