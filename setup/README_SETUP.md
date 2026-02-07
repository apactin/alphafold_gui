# First-Time Setup (WSL Ubuntu 22.04, CUDA 12.3.1)

This setup installs:
- Ubuntu 22.04 in WSL
- Miniconda + conda env `af3` (Python 3.11, CUDA 12.3)
- Docker Engine inside WSL + NVIDIA container toolkit (`docker --gpus all`)
- AlphaFold3 docker image build (`alphafold3`)
- MMseqs2 (via Linuxbrew)

> **Note:** I cannot distribute the AF3 weights. Request access from Google DeepMind: https://github.com/google-deepmind/alphafold3?tab=readme-ov-file

MMSeqs2 database is very large ~300 GB and must be downloaded manually.

---

## 0) Folder layout

This repo includes an installer folder:

setup/
README_SETUP.md
setup_windows.ps1
setup_wsl.sh
diskpart_expand_vhdx.dps


---

## 1) Install Ubuntu 22.04 (Windows)

Install from Microsoft Store:
- https://apps.microsoft.com/detail/9pn20msr04dw

Launch **Ubuntu 22.04** from the Windows Start Menu once it’s installed.

---

## 2) Optional: Expand the WSL disk (Windows)

If you need more disk space for databases/weights:

1. Open **PowerShell as Administrator**
2. Run:

```powershell
wsl --shutdown
diskpart 
select vdisk file="C:\Users\olive\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx" # replace with correct path
expand vidsk maximum=1500000 # 1.5 TB
wsl
```

Inside Ubuntu, find the device name and resize:

lsblk
sudo resize2fs /dev/sdX   # replace sdX with the correct device
df -h /


## 3) Run the WSL setup script (Ubuntu)
Inside Ubuntu:

```wsl
cd /mnt/c/Users/<YOU>/Documents/GitHub/<YOUR_REPO>
chmod +x install/setup_wsl.sh
./install/setup_wsl.sh
```

What it does:

Updates Ubuntu + installs base packages

Installs Miniconda (if missing) and creates conda env af3

Installs Docker Engine and configures GPU runtime

Clones AlphaFold3 to ~/Repositories/alphafold/alphafold3

Patches Dockerfile to add a venv with gemmi/pandas/requests

Builds docker image alphafold3

Installs mmseqs2 via Linuxbrew

Runs sanity checks

## 4) Manual downloads (required)
- AlphaFold3 weights
Place your weights here:

~/Repositories/alphafold/af_weights/af3.bin.zst
Then:

```
sudo apt install -y zstd
cd ~/Repositories/alphafold/af_weights
unzstd af3.bin.zst
```

- MMseqs database
Download uniref30_2302.db.tar.gz from:

https://opendata.mmseqs.org/colabfold

Copy into WSL and extract:

```
cp /mnt/c/Users/<YOU>/Downloads/uniref30_2302.db.tar.gz ~/Repositories/alphafold/mmseqs_db/
cd ~/Repositories/alphafold/mmseqs_db
tar -xzf uniref30_2302.db.tar.gz

export MMSEQS_NUM_THREADS=20
echo $MMSEQS_NUM_THREADS
```

5) AlphaFold3 patch note (manual)
AlphaFold uses Triton (optimizes for GPU). Depending on your GPU age, this may or may not be supported. If not, you will need to patch:

~/Repositories/alphafold/alphafold3/src/alphafold3/jax/attention/attention.py

Remove the final if (triton) block before the return statement.

6) Sanity checks

Check Docker GPU

```
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

Check AlphaFold3 container runs

```
docker run --rm -it \
  --gpus all \
  --volume ~/Repositories/alphafold/af_input:/work/af_input \
  --volume ~/Repositories/alphafold/af_output:/work/af_output \
  --volume ~/Repositories/alphafold/af_weights:/work/models \
  --volume ~/Repositories/alphafold/public_databases:/public_databases \
  alphafold3 python -c "print('container OK')"
  ```

MMseqs quick test
```
mmseqs easy-search <(echo -e ">q\nMKT...YOURSEQ") \
  ~/Repositories/alphafold/mmseqs_db/uniref30_2302_db outDB tmp && echo OK
  ```

7) Optional: Rosetta (minimization)
Download Rosetta (this version: https://downloads.rosettacommons.org/downloads/academic/2021/wk16/) and copy to WSL, e.g.:

```
mkdir -p ~/rosetta
cp /mnt/c/Users/<YOU>/Downloads/rosetta_bin_linux_2021.16.61629_bundle.tgz ~/rosetta/
cd ~/rosetta
tar -xvf rosetta_bin_linux_2021.16.61629_bundle.tgz
```

If you use LYX.params:

Copy LYX.params into Rosetta database:
.../database/chemical/residue_type_sets/fa_standard/residue_types/l-caa