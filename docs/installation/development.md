# Setting up the development environment

## Install Poppler

For MacOS, you need to run

```bash
brew install poppler
```

For Debian/Ubuntu, you need to run

```bash
sudo apt-get install -y poppler-utils
```

## Install the dependencies

Then, you can install the dependencies using uv in the virtual environment `.venv` using

```bash
git clone https://github.com/soumik12345/medrag-multi-modal
cd medrag-multi-modal
pip install -U pip uv
uv sync
```

After this, you need to activate the virtual environment using

```bash
source .venv/bin/activate
```

## [Optional] Install Flash Attention

In the activated virtual environment, you can optionally install Flash Attention (required for ColPali) using

```bash
uv pip install flash-attn --no-build-isolation
```