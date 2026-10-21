# Speculative decode — short index

Full write-up (FAST + HAC):

→ **[`2026-08-02-speculative-decode-report.md`](2026-08-02-speculative-decode-report.md)**

| Model | Headline | 20k wall | Δ median id |
|-------|----------|----------|-------------|
| **FAST** | q10_m2_brave | **≈9% faster** | ≈ −0.45 pp |
| **HAC** | q10_m2_brave | ≈0.6% faster | ≈ −0.18 pp |

```bash
./docs-riley/spec-fastq-sweep --fast
./docs-riley/spec-fastq-sweep --hac --quick
python3 fetch-and-minimap-spec.py --stamp <STAMP>
```
