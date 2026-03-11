#!/bin/bash
uvicorn api.main:app --host 0.0.0.0 --port 8002 &
sleep 3
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
