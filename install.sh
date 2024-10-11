#!/bin/bash

OS_TYPE=$(uname -s)

if [ "$OS_TYPE" = "Darwin" ]; then
    echo "Detected macOS."
    brew install poppler
elif [ "$OS_TYPE" = "Linux" ]; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then
            echo "Detected Ubuntu/Debian."
            sudo apt-get update
            sudo apt-get install -y poppler-utils
        else
            echo "Unsupported Linux distribution: $ID"
            exit 1
        fi
    else
        echo "Cannot detect Linux distribution."
        exit 1
    fi
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

git clone https://github.com/soumik12345/medrag-multi-modal
cd medrag-multi-modal
pip install -U .[core]
