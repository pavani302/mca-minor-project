"""Pneumonia detection from chest X-rays — source package.

Modules:
    config         – paths & hyperparameters (single source of truth)
    download_data  – fetch a subset of the Chest X-Ray Pneumonia dataset
    data_loader    – tf.data input pipelines + augmentation
    model          – DenseNet121 transfer-learning model
    gradcam        – Grad-CAM explainability
    train          – two-phase training entry point
    evaluate       – metrics + report figures
    predict        – single-image inference used by the CLI and the web app
"""
import os as _os

# macOS Python installs often lack system CA certificates, which breaks
# Keras' urllib download of the DenseNet121 ImageNet weights
# (SSL: CERTIFICATE_VERIFY_FAILED). Point SSL at certifi's bundle so the
# pretrained weights (and any other HTTPS fetch) download cleanly.
try:  # pragma: no cover - environment setup
    import certifi as _certifi

    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except Exception:
    pass
