#!/bin/bash
mkdir -p ~/.streamlit/
echo "[theme]
primaryColor = \"#4f7d86\"
backgroundColor = \"#ffffff\"
secondaryBackgroundColor = \"#f0f2f6\"
textColor = \"#262730\"
font = \"sans serif\"

[client]
showErrorDetails = true

[server]
port = 8501
headless = true
runOnSave = true
maxUploadSize = 200" > ~/.streamlit/config.toml
