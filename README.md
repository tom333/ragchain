https://github.com/ollama/ollama


kubectl port-forward deployment/ollama 3000:11434 -n datalab
kubectl port-forward svc/chroma-chromadb 8000:8000 -n datalab
