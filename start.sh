#!/usr/bin/env sh
cd "$(dirname "$0")" && exec streamlit run fem_gui/fem_app.py "$@"
