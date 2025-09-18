#!/bin/bash
set -e

DIRECTORIES=(
#    "./best_accuracy"
    "./best_ansatz"
    "./best_embedding"
    "./best_unsupervised_metric"
)

for DIR in "${DIRECTORIES[@]}"; do
    echo "=== Entrando em $DIR ==="
    cd "$DIR" || { echo "Falha ao acessar $DIR"; exit 1; }
    
    echo "▶️ Executando main_optimized.py em $PWD..."
    python main_optimized.py || { echo "❌ Falha em $DIR"; exit 1; }
    
    cd - > /dev/null  # Volta ao diretório anterior (silenciosamente)
    echo "✅ Concluído em $DIR"
done