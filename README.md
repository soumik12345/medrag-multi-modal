# MedRAG Multi-Modal

Multi-modal RAG for medical docmain.

## Installation

### For Development

For MacOS, you need to run

```bash
brew install poppler
```

For Debian/Ubuntu, you need to run

```bash
sudo apt-get install -y poppler-utils
```

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

In the activated virtual environment, you need to install Flash Attention using

```bash
uv pip install flash-attn --no-build-isolation
```
