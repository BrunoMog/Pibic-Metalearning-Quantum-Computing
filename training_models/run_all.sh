#!/bin/bash
#SBATCH --mem 16G
#SBATCH -c 16
#SBATCH -p long-simple
#SBATCH --gpus=0
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=basb@cin.ufpe.br

ENV_NAME=$1
module load Python3.10 Xvfb freeglut glew
python -m venv $HOME/doc/$ENV_NAME
source $HOME/doc/$ENV_NAME/bin/activate
which python
pip install -r ../requirements/requirements.txt
pip list

# Função para logging com timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Função para verificar se um diretório contém os experimentos esperados
check_experiment_structure() {
    local base_dir="$1"
    local missing_dirs=()
    
    for experiment in "best_accuracy" "best_ansatz" "best_embedding" "best_optimizer"; do
        if [[ ! -d "$base_dir/$experiment" ]]; then
            missing_dirs+=("$experiment")
        fi
    done
    
    if [[ ${#missing_dirs[@]} -gt 0 ]]; then
        log "⚠️  $base_dir está faltando: ${missing_dirs[*]}"
        return 1
    fi
    return 0
}

# Função para executar um experimento específico
run_experiment() {
    local experiment_dir="$1"
    local experiment_name="$2"
    
    log "▶️  Executando $experiment_name em $experiment_dir..."
    
    if [[ ! -f "$experiment_dir/main.py" ]]; then
        log "❌ Arquivo main.py não encontrado em $experiment_dir"
        return 1
    fi
    
    cd "$experiment_dir" || {
        log "❌ Falha ao acessar $experiment_dir"
        return 1
    }
    
    # Executar o experimento com timeout de 2 horas
    if timeout 7200 python main.py; then
        log "✅ $experiment_name concluído com sucesso"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log "⏰ $experiment_name interrompido por timeout (2h)"
        else
            log "❌ $experiment_name falhou com código $exit_code"
        fi
        cd - > /dev/null
        return $exit_code
    fi
    
    cd - > /dev/null
    return 0
}

# Encontrar todos os diretórios de experimentos
log "🔍 Procurando diretórios de experimentos..."

EXPERIMENT_DIRS=()
for dir in */; do
    dir="${dir%/}"  # Remove trailing slash
    
    # Verificar se é um diretório de experimento (contém os 4 subdiretórios)
    if check_experiment_structure "$dir"; then
        EXPERIMENT_DIRS+=("$dir")
        log "📁 Experimento encontrado: $dir"
    fi
done

# Verificar se encontrou experimentos
if [[ ${#EXPERIMENT_DIRS[@]} -eq 0 ]]; then
    log "❌ Nenhum diretório de experimento válido encontrado!"
    log "💡 Estrutura esperada: experiment_X/{best_accuracy,best_ansatz,best_embedding,best_unsupervised_metric}/"
    exit 1
fi

log "📊 Total de experimentos encontrados: ${#EXPERIMENT_DIRS[@]}"

# Tipos de experimentos para executar
EXPERIMENT_TYPES=(
    "best_accuracy"
    "best_ansatz" 
    "best_embedding"
    "best_unsupervised_metric"
)

# Contadores para estatísticas
total_experiments=0
successful_experiments=0
failed_experiments=0
start_time=$(date +%s)

# Executar todos os experimentos
for experiment_dir in "${EXPERIMENT_DIRS[@]}"; do
    log "🚀 === Processando experimento: $experiment_dir ==="
    
    for experiment_type in "${EXPERIMENT_TYPES[@]}"; do
        full_path="$experiment_dir/$experiment_type"
        
        if [[ -d "$full_path" ]]; then
            total_experiments=$((total_experiments + 1))
            
            if run_experiment "$full_path" "$experiment_dir/$experiment_type"; then
                successful_experiments=$((successful_experiments + 1))
            else
                failed_experiments=$((failed_experiments + 1))
                log "⚠️  Continuando com próximo experimento..."
            fi
        else
            log "⚠️  Diretório $full_path não encontrado, pulando..."
        fi
    done
    
    log "✅ Experimento $experiment_dir processado"
    echo ""
done

# Calcular tempo total
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

# Relatório final
log "🎯 === RELATÓRIO FINAL ==="
log "📊 Total de experimentos: $total_experiments"
log "✅ Sucessos: $successful_experiments"
log "❌ Falhas: $failed_experiments"
log "⏱️  Tempo total: ${hours}h ${minutes}m ${seconds}s"

if [[ $failed_experiments -eq 0 ]]; then
    log "🎉 Todos os experimentos executados com sucesso!"
    exit 0
else
    log "⚠️  Alguns experimentos falharam. Verifique os logs acima."
    exit 1
fi