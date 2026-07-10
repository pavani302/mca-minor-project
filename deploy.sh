#!/usr/bin/env bash
#
# deploy.sh — one-shot setup + run for the Pneumonia Detection project.
#
# Designed for a fresh Ubuntu machine (e.g. an AWS EC2 instance). After you
# clone the repo, run this script and it will:
#   1. install system dependencies (Python, pip, venv, git)
#   2. create a virtualenv and install the Python requirements
#   3. ask you which dataset to use (small sample or the full dataset)
#   4. download the data and train the model (unless a model already exists)
#   5. launch the Streamlit web app, reachable from your browser
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh                 # interactive
#   DATASET=sample ./deploy.sh  # non-interactive: small subset
#   DATASET=full   ./deploy.sh  # non-interactive: full dataset
#   SKIP_RUN=1     ./deploy.sh  # set up + train but don't launch the app
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Pretty logging
# --------------------------------------------------------------------------- #
BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"; RED="\033[31m"; RESET="\033[0m"
log()  { echo -e "${BLUE}${BOLD}==>${RESET} ${BOLD}$*${RESET}"; }
ok()   { echo -e "${GREEN}  ✓ $*${RESET}"; }
warn() { echo -e "${YELLOW}  ! $*${RESET}"; }
err()  { echo -e "${RED}  ✗ $*${RESET}" >&2; }

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
PORT="${PORT:-8501}"

log "Pneumonia Detection — deployment on $(uname -s) ($(uname -m))"
echo "    Project directory: ${PROJECT_DIR}"

# --------------------------------------------------------------------------- #
# 1. System dependencies (Debian/Ubuntu)
# --------------------------------------------------------------------------- #
install_system_deps() {
    log "Step 1/5 — Installing system dependencies"
    if command -v apt-get >/dev/null 2>&1; then
        SUDO=""
        [ "$(id -u)" -ne 0 ] && SUDO="sudo"
        $SUDO apt-get update -y
        # python3-venv/pip for the environment; git to clone/update; curl for health check.
        # opencv is installed as the *headless* wheel, so no libGL/X11 packages are needed.
        $SUDO apt-get install -y python3 python3-venv python3-pip git curl
        ok "System packages installed"
    else
        warn "apt-get not found — assuming Python 3 and pip are already available."
        command -v python3 >/dev/null 2>&1 || { err "python3 is required but not installed."; exit 1; }
    fi
}

# --------------------------------------------------------------------------- #
# 2. Python virtualenv + requirements
# --------------------------------------------------------------------------- #
setup_venv() {
    log "Step 2/5 — Creating virtualenv and installing Python packages"
    if [ ! -d .venv ]; then
        python3 -m venv .venv
        ok "Created .venv"
    else
        ok "Reusing existing .venv"
    fi
    # shellcheck disable=SC1091
    ./.venv/bin/python -m pip install --quiet --upgrade pip
    ./.venv/bin/python -m pip install --quiet -r requirements.txt
    ok "Python dependencies installed"
}

# --------------------------------------------------------------------------- #
# 3. Choose the dataset
# --------------------------------------------------------------------------- #
choose_dataset() {
    log "Step 3/5 — Dataset selection"
    if [ -n "${DATASET:-}" ]; then
        CHOICE="$DATASET"
        ok "Using DATASET=${CHOICE} from environment"
    else
        echo ""
        echo "    Which dataset do you want to use?"
        echo -e "      ${BOLD}1${RESET}) Small sample  — ~800 balanced images. Fast (a few minutes). Good for a quick demo."
        echo -e "      ${BOLD}2${RESET}) Full dataset  — the complete Chest X-Ray Pneumonia set. Slower, higher accuracy."
        echo ""
        read -r -p "    Enter 1 or 2 [1]: " ANS
        case "${ANS:-1}" in
            2) CHOICE="full" ;;
            *) CHOICE="sample" ;;
        esac
    fi

    if [ "$CHOICE" = "full" ]; then
        DOWNLOAD_ARGS="--full"
        ok "Selected: FULL dataset"
    else
        DOWNLOAD_ARGS=""
        ok "Selected: SMALL sample subset"
    fi
}

# --------------------------------------------------------------------------- #
# 4. Download data + train
# --------------------------------------------------------------------------- #
download_and_train() {
    log "Step 4/5 — Preparing data and model"

    if [ -f models/pneumonia_densenet121.keras ] && [ -z "${FORCE_TRAIN:-}" ]; then
        warn "A trained model already exists (models/pneumonia_densenet121.keras)."
        warn "Skipping download + training. Set FORCE_TRAIN=1 to retrain from scratch."
        return
    fi

    echo "    Downloading dataset ${DOWNLOAD_ARGS:-(sample)} ..."
    ./.venv/bin/python -m src.download_data $DOWNLOAD_ARGS
    ok "Dataset ready"

    echo "    Training the model (this can take a while on a CPU-only instance) ..."
    ./.venv/bin/python -m src.train
    ok "Model trained and evaluated"
}

# --------------------------------------------------------------------------- #
# 5. Launch the web app
# --------------------------------------------------------------------------- #
run_app() {
    if [ -n "${SKIP_RUN:-}" ]; then
        log "Step 5/5 — Skipping app launch (SKIP_RUN set)"
        echo "    To start it later:"
        echo "      ./.venv/bin/streamlit run app/Home.py --server.address 0.0.0.0 --server.port ${PORT}"
        return
    fi

    log "Step 5/5 — Launching the Streamlit web app"
    PUBLIC_IP="$(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
    echo ""
    ok "Starting on port ${PORT} (bound to 0.0.0.0)"
    if [ -n "$PUBLIC_IP" ]; then
        echo -e "    ${BOLD}Open:${RESET} http://${PUBLIC_IP}:${PORT}"
        warn "Make sure your EC2 security group allows inbound TCP ${PORT}."
    else
        echo -e "    ${BOLD}Open:${RESET} http://<this-machine-ip>:${PORT}"
    fi
    echo ""
    exec ./.venv/bin/streamlit run app/Home.py \
        --server.address 0.0.0.0 \
        --server.port "${PORT}" \
        --server.headless true \
        --browser.gatherUsageStats false
}

# --------------------------------------------------------------------------- #
main() {
    install_system_deps
    setup_venv
    choose_dataset
    download_and_train
    run_app
}

main "$@"
