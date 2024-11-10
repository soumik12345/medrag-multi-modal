# MedRAG Multi-Modal

[![Documentation](https://img.shields.io/badge/MedRaG-docs-blue)](https://geekyrakshit.dev/medrag-multi-modal)

An ongoing journey to build a robust and scaleable multi-modal question answering system for the domain of medicine and life sciences powered by SoTA generative AI, NLP, and computer vision models.

## Installation

### Install Poppler

To convert PDFs to images, we utilize the `pdf2image` library, which requires `poppler` to be installed on your system.

#### MacOS with homebrew

```bash
brew install poppler
```

#### Debian/Ubuntu

```
sudo apt-get install -y poppler-utils
```

### Install MedRAG Multi-Modal library

```bash
git clone https://github.com/soumik12345/medrag-multi-modal
cd medrag-multi-modal
pip install -e .
```

You can also install the library using `uv` by running

```bash
pip install -U pip uv
uv pip install .
source .venv/bin/activate
```

We also provide an experimental shell script to install the dependencies and the library.

```bash
git clone https://github.com/soumik12345/medrag-multi-modal
cd medrag-multi-modal
sh install.sh # Add --flash-attention to install flash-attn with no build isolation.
```
