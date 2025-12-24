apt update -y && apt install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libffi-dev \
  liblzma-dev \
  tk-dev \
  wget curl llvm

echo "install python 3.10"
export PYENV_ROOT=/workspace/.pyenv
export PATH="$PYENV_ROOT/bin:$PATH"
curl -fsSL https://pyenv.run | bash

pyenv install 3.10
pyenv local 3.10
pipx install virtualenv

