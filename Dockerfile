# hash:sha256:fec854bf69225e1b3a5df2befad3a55b45ad105fd9c1635b5a4f7db628d53a01
FROM registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    pandas==2.2.3 \
    rdkit==2024.9.6 \
    scikit-learn==1.6.1 \
    scikit-multilearn==0.2.0 \
    torch-geometric==2.6.1
