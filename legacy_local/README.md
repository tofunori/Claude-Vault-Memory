# Legacy Local (Claude-only)

Ce dossier contient l'ancien pipeline mémoire local (pré-NAS unifié).
Nom historique dans cette archive: **Claude Vault Memory**.

## Contenu

- `enqueue.py`: ancien hook Stop local
- `vault_session_brief.py`: ancien hook SessionStart local
- `vault_status.py`: dashboard local
- `vault_reflect.py`: maintenance locale périodique
- `install.sh`: installateur local historique
- `launchd/`: service launchd local macOS
- `README_LOCAL_LEGACY.md`: documentation historique complète

## Statut

- Maintenu pour référence/migration uniquement.
- Considéré read-only (pas de développement actif).
- Non recommandé pour la prod actuelle.
- Le mode recommandé est la stack NAS sous `../nas_memory/`.
