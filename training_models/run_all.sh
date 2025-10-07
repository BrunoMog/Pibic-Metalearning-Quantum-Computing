#!/bin/bash
#SBATCH --mem 16G
#SBATCH -c 16
#SBATCH -p long-simple
#SBATCH --gpus=0
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=basb@cin.ufpe.br

# Verificar se foi passado nome do ambiente
if [[ -z "$1" ]]; then
    echo "❌ Erro: Nome do ambiente virtual não fornecido!"
    echo "💡 Uso: sbatch run_all.sh <nome_do_ambiente>"
    exit 1
fi

ENV_NAME=$1
echo "🐍 Configurando ambiente virtual: $ENV_NAME"

# Carregar módulos
module load Python3.10 Xvfb freeglut glew

# Criar ambiente virtual
echo "📦 Criando ambiente virtual..."
python -m venv $HOME/doc/$ENV_NAME

# Ativar ambiente virtual e verificar
echo "🔌 Ativando ambiente virtual..."
source $HOME/doc/$ENV_NAME/bin/activate

# Verificar se a ativação funcionou
echo "🔍 Verificando ativação do ambiente:"
which python
python --version
echo "VIRTUAL_ENV: $VIRTUAL_ENV"

# Verificar se o arquivo requirements existe
if [[ ! -f "../requirements/requirements.txt" ]]; then
    echo "❌ Arquivo requirements.txt não encontrado em ../requirements/requirements.txt"
    echo "📁 Arquivos disponíveis em ../requirements/:"
    ls -la ../requirements/ || echo "Diretório ../requirements/ não existe"
    exit 1
fi

# Atualizar pip e instalar dependências
echo "📦 Atualizando pip..."
pip install --upgrade pip

echo "📦 Instalando dependências..."
if pip install -r ../requirements/requirements.txt; then
    echo "✅ Dependências instaladas com sucesso"
else
    echo "❌ Falha na instalação das dependências"
    exit 1
fi

# Verificar se as bibliotecas essenciais foram instaladas
echo "🔍 Verificando bibliotecas instaladas:"
python -c "import numpy; print(f'✅ numpy: {numpy.__version__}')" || { echo "❌ numpy não instalado corretamente"; exit 1; }
python -c "import pandas; print(f'✅ pandas: {pandas.__version__}')" || { echo "❌ pandas não instalado corretamente"; exit 1; }
python -c "import sklearn; print(f'✅ scikit-learn: {sklearn.__version__}')" || { echo "❌ scikit-learn não instalado corretamente"; exit 1; }
python -c "import joblib; print(f'✅ joblib: {joblib.__version__}')" || { echo "❌ joblib não instalado corretamente"; exit 1; }

echo "🎉 Ambiente configurado com sucesso!"

# Função para logging com timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Função para verificar se um diretório contém os experimentos esperados
check_experiment_structure() {
    local base_dir="$1"
    local missing_dirs=()
    
    # ESTRUTURA CORRETA (sem best_unsupervised_metric)
    for experiment in "best_accuracy" "best_ansatz" "best_embedding"; do
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
    
    log "▶️  Executando $experiment_name..."
    
    if [[ ! -f "$experiment_dir/main.py" ]]; then
        log "❌ Arquivo main.py não encontrado em $experiment_dir"
        return 1
    fi
    
    cd "$experiment_dir" || {
        log "❌ Falha ao acessar $experiment_dir"
        return 1
    }
    
    # Garantir que o ambiente virtual está ativo
    source $HOME/doc/$ENV_NAME/bin/activate
    
    # Verificar bibliotecas antes de executar
    if ! python -c "import numpy, pandas, sklearn, joblib" 2>/dev/null; then
        log "❌ Bibliotecas não disponíveis no ambiente - reativando..."
        source $HOME/doc/$ENV_NAME/bin/activate
        if ! python -c "import numpy, pandas, sklearn, joblib" 2>/dev/null; then
            log "❌ Falha crítica: bibliotecas não encontradas"
            cd - > /dev/null
            return 1
        fi
    fi
    
    # Executar o experimento com timeout de 4 horas
    if timeout 14400 python main.py; then
        log "✅ $experiment_name concluído com sucesso"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log "⏰ $experiment_name interrompido por timeout (4h)"
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
    
    # Pular arquivos Python
    if [[ "$dir" == *".py" ]]; then
        continue
    fi
    
    # Verificar se é um diretório de experimento
    if check_experiment_structure "$dir"; then
        EXPERIMENT_DIRS+=("$dir")
        log "📁 Experimento encontrado: $dir"
    else
        log "⚠️  $dir não possui estrutura de experimento completa"
    fi
done

# Verificar se encontrou experimentos
if [[ ${#EXPERIMENT_DIRS[@]} -eq 0 ]]; then
    log "❌ Nenhum diretório de experimento válido encontrado!"
    log "💡 Estrutura esperada: experiment_X/{best_accuracy,best_ansatz,best_embedding}/"
    exit 1
fi

log "📊 Total de experimentos encontrados: ${#EXPERIMENT_DIRS[@]}"

# TIPOS DE EXPERIMENTOS CORRIGIDOS (sem best_unsupervised_metric)
EXPERIMENT_TYPES=(
    "best_accuracy"
    "best_ansatz" 
    "best_embedding"
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