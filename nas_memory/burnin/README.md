# Burn-in 72h (NAS memory)

Ce dossier contient les outils pour exécuter et évaluer un burn-in 72h:

- erreurs queue
- latence `/retrieve`
- taux promotion/bruit sur mémoire live
- réduction des nœuds mémoire isolés (V1.3R)
- verdict automatique `PASS/FAIL` (gate strict)

## Lancer un burn-in 72h

Depuis `/volume1/Services/memory` (NAS):

```bash
python3 nas_memory/burnin/collector.py \
  --duration-hours 72 \
  --mode mixed \
  --gate strict
```

Artefacts produits:

- `/volume1/Services/memory/state/burnin/<run_id>/config.json`
- `/volume1/Services/memory/state/burnin/<run_id>/health_samples.jsonl`
- `/volume1/Services/memory/state/burnin/<run_id>/retrieve_samples.jsonl`
- `/volume1/Services/memory/state/burnin/<run_id>/worker_live_samples.jsonl`
- `/volume1/Services/memory/state/burnin/<run_id>/memory_action_samples.jsonl`
- `/volume1/Services/memory/state/burnin/<run_id>/synthetic_trace.jsonl`
- `/volume1/Services/memory/state/burnin/<run_id>/graph_samples.jsonl`
- `/volume1/Services/memory/state/burnin/<run_id>/relation_stats_samples.jsonl`
- `/volume1/Services/memory/state/burnin/<run_id>/summary.json`
- `/volume1/Services/memory/state/burnin/<run_id>/summary.md`
- `/volume1/Services/memory/state/burnin/<run_id>/passfail.json`

## Pré-vol 30 minutes

```bash
python3 nas_memory/burnin/collector.py \
  --duration-hours 0.5 \
  --mode mixed \
  --gate strict
```

## Générer/recalculer le rapport

```bash
python3 nas_memory/burnin/report.py \
  --run-dir /volume1/Services/memory/state/burnin/<run_id> \
  --gate strict
```

## Seuils stricts

- `queue_error_rate <= 0.005`
- `queue_backlog_stall_minutes <= 10`
- `retrieve_p95_ms <= retrieve_p95_baseline_ms * 1.20` (fallback `<= 500` si baseline manquante)
- `retrieve_p99_ms <= 1200`
- `retrieve_error_rate <= 0.01`
- `promotion_rate >= 0.20`
- `noise_rate <= 0.80`
- `session_leak_count == 0`
- `memory_singleton_reduction >= 0.40`
- `relation_precision_sample >= 0.85` (optionnel via `relation_precision_manual.json`, sinon neutre)
