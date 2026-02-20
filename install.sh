#!/bin/bash
# install.sh — Installation interactive de claude-vault-memory
# Usage: bash install.sh

set -e

echo "=== claude-vault-memory — Installation ==="
echo ""

# 1. Dépendances Python
echo "[1/4] Installation des packages Python..."
pip3 install cohere qdrant-client openai --break-system-packages 2>/dev/null || \
pip3 install cohere qdrant-client openai

echo "  ✓ cohere, qdrant-client, openai installés"
echo ""

# 2. Config
if [ ! -f config.py ]; then
    echo "[2/4] Création de config.py depuis config.example.py..."
    cp config.example.py config.py
    echo "  → Éditez config.py avec vos chemins et clés avant de continuer"
    echo "  → Puis relancez: bash install.sh"
    exit 0
else
    echo "[2/4] config.py trouvé ✓"
fi

# 3. Clé Cohere
echo ""
echo "[3/4] Configuration COHERE_API_KEY"
echo "  Obtenez votre clé sur: https://dashboard.cohere.com/api-keys"
read -p "  COHERE_API_KEY (laisser vide pour configurer manuellement dans .env): " cohere_key
if [ -n "$cohere_key" ]; then
    ENV_FILE=$(python3 -c "from config import ENV_FILE; print(ENV_FILE)" 2>/dev/null || echo "$HOME/.claude/hooks/.env")
    if grep -q "COHERE_API_KEY" "$ENV_FILE" 2>/dev/null; then
        sed -i.bak "s|COHERE_API_KEY=.*|COHERE_API_KEY=$cohere_key|" "$ENV_FILE"
    else
        echo "COHERE_API_KEY=$cohere_key" >> "$ENV_FILE"
    fi
    echo "  ✓ Clé ajoutée dans $ENV_FILE"
fi

# 4. Build index initial
echo ""
echo "[4/4] Build de l'index vectoriel initial..."
python3 vault_embed.py
echo ""
echo "=== Installation terminée ==="
echo ""
echo "Prochaines étapes :"
echo "  1. Ajouter vault_retrieve.py dans UserPromptSubmit de settings.json"
echo "  2. Configurer le launchd plist (voir launchd/com.example.vault-queue-worker.plist)"
echo "  3. Tester: echo '{\"prompt\":\"votre question\"}' | python3 vault_retrieve.py"
