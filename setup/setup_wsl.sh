#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# setup_wsl.sh — WSL (Ubuntu 22.04) setup for AF3 + Docker GPU + MMseqs2
# Matches your guide but makes it idempotent and ordered correctly.
# ============================================================

log() { echo -e "\n\033[1;32m==>\033[0m $*"; }
warn() { echo -e "\n\033[1;33m[warn]\033[0m $*"; }
die() { echo -e "\n\033[1;31m[err]\033[0m $*"; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1; }

UBUNTU_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"
REPO_ROOT="$HOME/Repositories/alphafold"
AF3_DIR="$REPO_ROOT/alphafold3"

# You can override these at runtime:
#   CONDA_PREFIX_ROOT=...  (where to install miniconda)
#   AF3_GIT_URL=...        (if you forked)
CONDA_PREFIX_ROOT="${CONDA_PREFIX_ROOT:-$HOME/miniconda3}"
AF3_GIT_URL="${AF3_GIT_URL:-https://github.com/google-deepmind/alphafold3.git}"

# ---------- helpers ----------
ensure_python_is_python3() {
  if need_cmd python; then
    return 0
  fi
  log "Making 'python' resolve to python3 (python-is-python3)"
  sudo apt-get update -y
  sudo apt-get install -y python-is-python3
}

apt_update_upgrade() {
  log "apt update + full-upgrade"
  sudo apt-get update -y
  sudo apt-get -y full-upgrade
}

install_base_packages() {
  log "Installing base Ubuntu packages"
  sudo apt-get install -y \
    build-essential git curl wget unzip tar pigz \
    cmake pkg-config ca-certificates \
    python3 python3-pip python3-venv \
    jq zstd \
    openbabel \
    libglib2.0-0 libxrender1 libxext6 libsm6 \
    scons python3-dev zlib1g-dev libbz2-dev \
    procps file
}

install_miniconda_if_needed() {
  if [[ -x "$CONDA_PREFIX_ROOT/bin/conda" ]]; then
    log "Miniconda already present at: $CONDA_PREFIX_ROOT"
    return 0
  fi

  log "Installing Miniconda to: $CONDA_PREFIX_ROOT"
  local installer="Miniconda3-latest-Linux-x86_64.sh"
  wget -q "https://repo.anaconda.com/miniconda/${installer}" -O "/tmp/${installer}"
  bash "/tmp/${installer}" -b -p "$CONDA_PREFIX_ROOT"

  # Activate conda for this shell + future shells
  "$CONDA_PREFIX_ROOT/bin/conda" init bash >/dev/null
  # shellcheck disable=SC1090
  source "$CONDA_PREFIX_ROOT/etc/profile.d/conda.sh"
}

ensure_conda_loaded() {
  if need_cmd conda; then
    return 0
  fi
  if [[ -f "$CONDA_PREFIX_ROOT/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_PREFIX_ROOT/etc/profile.d/conda.sh"
  fi
  need_cmd conda || die "conda not found. Open a new shell or source conda.sh."
}

create_conda_env_af3() {
  log "Creating/ensuring conda env: af3 (python=3.11, CUDA 12.3)"
  ensure_conda_loaded

  if conda env list | awk '{print $1}' | grep -qx "af3"; then
    log "Conda env 'af3' already exists"
  else
    conda create -y -n af3 python=3.11 -c conda-forge cuda-nvcc=12.3 cuda-cudart-dev=12.3
  fi

  # shellcheck disable=SC1090
  source "$CONDA_PREFIX_ROOT/etc/profile.d/conda.sh"
  conda activate af3

  log "Installing conda packages (rdkit, gemmi, pandas, etc.)"
  conda install -y -c conda-forge \
    setuptools pandas scipy biopython gemmi rdkit zlib bzip2 rust requests colorama absl-py

  # You had both conda-forge + nvidia installs; keep nvidia toolkit separate.
  log "Installing CUDA toolkit from nvidia channel (inside conda env)"
  conda install -y -c nvidia cuda-toolkit

  log "Installing pip packages"
  pip install --upgrade pip
  pip install "numpy<2" pybel

  python3 -c "import pandas,gemmi; print('pandas',pandas.__version__); print('gemmi OK')"
}

setup_git_defaults() {
  log "Configuring git defaults"
  git config --global pull.rebase false
}

install_docker_engine() {
  if need_cmd docker; then
    log "Docker already installed: $(docker --version)"
    return 0
  fi

  log "Installing Docker Engine from Docker apt repo"
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME} stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  log "Adding user to docker group (may require a new shell to fully apply)"
  sudo usermod -aG docker "$USER" || true
}

docker_hello_world() {
  log "Testing docker hello-world"
  # If group membership not active yet, this may fail; fall back to sudo.
  if docker run --rm hello-world >/dev/null 2>&1; then
    log "docker hello-world OK (no sudo)"
  else
    warn "docker without sudo failed (group may not be active). Trying with sudo..."
    sudo docker run --rm hello-world
    warn "To use docker without sudo, open a new terminal or run: newgrp docker"
  fi
}

install_nvidia_container_toolkit() {
  log "Installing NVIDIA container toolkit (for docker --gpus)"
  distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  sudo apt-get update -y
  sudo apt-get install -y nvidia-container-toolkit

  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker || true
}

configure_docker_default_runtime_nvidia() {
  log "Configuring Docker daemon.json default-runtime=nvidia"
  local daemon="/etc/docker/daemon.json"
  sudo mkdir -p /etc/docker

  if [[ ! -f "$daemon" ]]; then
    # Minimal config with runtimes block
    sudo tee "$daemon" >/dev/null <<'JSON'
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
JSON
  else
    # If file exists, do a conservative edit: if default-runtime missing, add it.
    if ! sudo jq -e '.["default-runtime"]' "$daemon" >/dev/null 2>&1; then
      sudo jq '. + {"default-runtime":"nvidia"}' "$daemon" | sudo tee "$daemon" >/dev/null
    fi
    # Ensure runtimes.nvidia exists
    if ! sudo jq -e '.runtimes.nvidia' "$daemon" >/dev/null 2>&1; then
      sudo jq '. + {"runtimes": ( .runtimes // {} )}' "$daemon" | sudo tee "$daemon" >/dev/null
      sudo jq '.runtimes += {"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}}' "$daemon" | sudo tee "$daemon" >/dev/null
    fi
  fi

  sudo systemctl restart docker || true
  docker info | grep -i runtime || true
}

test_docker_gpu() {
  log "Testing docker GPU access (nvidia-smi in container)"
  if docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi; then
    log "GPU container OK"
  else
    warn "GPU container test failed. Common causes:"
    warn " - Windows NVIDIA driver not installed / not recent"
    warn " - WSL GPU support not enabled"
    warn " - You need to restart WSL: in PowerShell => wsl --shutdown"
  fi
}

create_project_dirs() {
  log "Creating project directories under: $REPO_ROOT"
  mkdir -p "$REPO_ROOT"/{af_input,af_output,af_weights,public_databases,mmseqs_db,msa_cache,ligands}
  log "Dirs:"
  ls -ld "$REPO_ROOT"/{af_input,af_output,af_weights,public_databases,mmseqs_db,msa_cache,ligands}
}

clone_alphafold3() {
  if [[ -d "$AF3_DIR/.git" ]]; then
    log "alphafold3 already cloned at: $AF3_DIR"
    return 0
  fi
  log "Cloning AlphaFold3 repo"
  git clone "$AF3_GIT_URL" "$AF3_DIR"
}

patch_attention_py() {
  # You said: "remove final if(triton) block before the return statement"
  # We can't perfectly infer without exact file contents; do a safe, manual-marker approach:
  local f="$AF3_DIR/src/alphafold3/jax/attention/attention.py"
  [[ -f "$f" ]] || { warn "attention.py not found at expected path: $f"; return 0; }

  log "NOTE: attention.py patch is highly version-dependent."
  warn "I did NOT auto-delete code here (too risky without exact snippet)."
  warn "Open: $f"
  warn "Find: the final 'if triton' block immediately before the return, and remove that block."
}

patch_dockerfile_add_gemmi() {
  local df="$AF3_DIR/docker/Dockerfile"
  [[ -f "$df" ]] || { warn "Dockerfile not found: $df"; return 0; }

  if grep -q "pip install gemmi pandas requests" "$df"; then
    log "Dockerfile already contains gemmi/pandas/requests venv block"
    return 0
  fi

  log "Patching Dockerfile to add venv + gemmi/pandas/requests"
  # Insert after the first FROM line (simple + usually safe)
  awk '
    NR==1 {print; next}
    NR==2 {
      print
      print ""
      print "RUN python3 -m venv /venv \\"
      print "    && /venv/bin/pip install --upgrade pip \\"
      print "    && /venv/bin/pip install gemmi pandas requests"
      print "ENV PATH=\"/venv/bin:$PATH\""
      next
    }
    {print}
  ' "$df" > /tmp/Dockerfile.patched

  sudo cp "$df" "$df.bak.$(date +%Y%m%d_%H%M%S)"
  sudo cp /tmp/Dockerfile.patched "$df"
}

build_af3_docker() {
  log "Building AlphaFold3 docker image (alphafold3)"
  cd "$AF3_DIR"
  docker build --network=host -f docker/Dockerfile -t alphafold3 .
}

install_homebrew_mmseqs2() {
  if need_cmd mmseqs; then
    log "mmseqs2 already installed: $(mmseqs --version 2>/dev/null || echo OK)"
    return 0
  fi

  log "Installing Homebrew (Linuxbrew) + mmseqs2"
  if [[ ! -d "/home/linuxbrew/.linuxbrew" ]]; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi

  if ! grep -q 'linuxbrew' "$HOME/.profile" 2>/dev/null; then
    echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> "$HOME/.profile"
  fi

  eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
  brew install mmseqs2
}

mmseqs_db_instructions() {
  log "MMseqs DB: you must download uniref30_2302.db.tar.gz manually (large)"
  echo "Place it here:"
  echo "  $REPO_ROOT/mmseqs_db/uniref30_2302.db.tar.gz"
  echo ""
  echo "Then run:"
  echo "  cd $REPO_ROOT/mmseqs_db"
  echo "  tar -xzf uniref30_2302.db.tar.gz"
  echo "  export MMSEQS_NUM_THREADS=20"
  echo "  echo \$MMSEQS_NUM_THREADS"
}

sanity_checks() {
  log "Sanity checks: folders"
  ls "$REPO_ROOT/af_input" "$REPO_ROOT/af_output" "$REPO_ROOT/af_weights" "$REPO_ROOT/public_databases" >/dev/null

  log "Sanity check: container python"
  docker run --rm -it \
    --gpus all \
    --volume "$REPO_ROOT/af_input:/work/af_input" \
    --volume "$REPO_ROOT/af_output:/work/af_output" \
    --volume "$REPO_ROOT/af_weights:/work/models" \
    --volume "$REPO_ROOT/public_databases:/public_databases" \
    alphafold3 python -c "print('container OK')"
}

print_next_steps() {
  echo ""
  log "NEXT STEPS (manual parts)"
  echo "1) Put your weights:  af3.bin.zst  -> $REPO_ROOT/af_weights"
  echo "   Then: unzstd $REPO_ROOT/af_weights/af3.bin.zst"
  echo ""
  echo "2) Download MMseqs DB uniref30_2302.db.tar.gz and extract into:"
  echo "   $REPO_ROOT/mmseqs_db"
  echo ""
  echo "3) Patch attention.py manually (version dependent):"
  echo "   $AF3_DIR/src/alphafold3/jax/attention/attention.py"
  echo ""
  echo "4) If GPU test failed, run in PowerShell:"
  echo "   wsl --shutdown"
  echo "   then reopen Ubuntu and re-run: docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi"
  echo ""
}

main() {
  [[ -f /etc/os-release ]] || die "This script must be run inside Ubuntu/WSL."

  apt_update_upgrade
  install_base_packages
  ensure_python_is_python3

  setup_git_defaults
  install_miniconda_if_needed
  create_conda_env_af3

  install_docker_engine
  docker_hello_world

  install_nvidia_container_toolkit
  configure_docker_default_runtime_nvidia
  test_docker_gpu

  create_project_dirs
  clone_alphafold3
  patch_attention_py
  patch_dockerfile_add_gemmi
  build_af3_docker

  install_homebrew_mmseqs2
  mmseqs_db_instructions

  sanity_checks
  print_next_steps

  log "WSL setup completed."
}

main "$@"
