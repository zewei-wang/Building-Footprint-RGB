#!/usr/bin/env bash
#
# Provision a pyenv-based environment for the AWS open data satellite/LiDAR tutorial.
# The script targets Ubuntu hosts with pyenv and pyenv-virtualenv already installed.

set -euo pipefail

PYTHON_VERSION=${PYTHON_VERSION:-3.8.18}
ENV_NAME=${1:-lidar-tutorial}
REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-requirements-pyenv.txt}
DISPLAY_NAME=${DISPLAY_NAME:-"Lidar Tutorial (pyenv)"}

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REQ_PATH="${PROJECT_ROOT}/${REQUIREMENTS_FILE}"

if [[ ! -f "${REQ_PATH}" ]]; then
  echo "Expected requirements file '${REQ_PATH}' was not found." >&2
  exit 1
fi

if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "This helper is tailored for Ubuntu. Detected ID='${ID:-unknown}'." >&2
    exit 1
  fi
fi

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv is not on PATH. Install pyenv before running this script." >&2
  exit 1
fi

if ! pyenv commands | grep -q "^virtualenv$"; then
  echo "pyenv-virtualenv plugin is required (pyenv command 'virtualenv' not found)." >&2
  exit 1
fi

SUDO=sudo
if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  SUDO=
fi

APT_PACKAGES=(
  build-essential
  curl
  git
  libbz2-dev
  libffi-dev
  libgdal-dev
  libgeos-dev
  liblzma-dev
  libncursesw5-dev
  libproj-dev
  libreadline-dev
  libsqlite3-dev
  libssl-dev
  libxml2-dev
  libxmlsec1-dev
  libspatialindex-dev
  tk-dev
  uuid-dev
  xz-utils
  zlib1g-dev
  gdal-bin
)

echo "Installing system dependencies via apt..."
$SUDO apt-get update
$SUDO apt-get install -y "${APT_PACKAGES[@]}"

echo "Configuring pyenv for Python ${PYTHON_VERSION} (${ENV_NAME})..."
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv install -s "${PYTHON_VERSION}"

if ! pyenv virtualenvs --bare | grep -Fxq "${ENV_NAME}"; then
  pyenv virtualenv "${PYTHON_VERSION}" "${ENV_NAME}"
else
  echo "pyenv virtualenv '${ENV_NAME}' already exists; reusing it."
fi

pyenv shell "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_PATH}"

python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${DISPLAY_NAME}"

cat <<EOF

Done!

Environment: ${ENV_NAME}
Python     : ${PYTHON_VERSION}
Kernel     : ${DISPLAY_NAME}

To activate in this shell: pyenv activate ${ENV_NAME}
To make the project default: pyenv local ${ENV_NAME}

EOF
