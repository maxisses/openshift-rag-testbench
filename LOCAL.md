## install nvidia driver 
## install container toolkit
## run cdi configuration correctly (nvidia-ctk cdi list)
## provide read access to cdi configuration: chmod + ... /etc/cdi/nvidia.yaml

## create common network
podman network create rag_network

## start VLLM container locally
podman run -d --rm --name vllm --security-opt label=disable --device nvidia.com/gpu=all \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/vllm/vllm-native/localdata:/root/.cache/huggingface \
    -e HF_TOKEN=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 \
    -p 8000:8000 \
    --network rag_network \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.2 --max-model-len 512

## start OLLAMA container locally
podman run -it --rm --name ollama --security-opt label=disable --device nvidia.com/gpu=all \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/ollama/localdata:/root/.ollama \
    -e HF_TOKEN=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 \
    -p 11434:11434 \
    --network rag_network \
    ollama/ollama:latest

## start jupyter locally
podman run -p 8888:8888 -d --rm \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/notebooks-for-ingestion/localdata:/opt/app-root/src/mydata \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/notebooks-for-ingestion/localdata/huggingface:/opt/app-root/src/.cache/ \
    --security-opt label=disable --device nvidia.com/gpu=all -e HF_TOKEN=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 \
    --network rag_network \
    quay.io/mdargatz/rag-ingest-workbench

## start milvus locally
podman run -d --rm \
    --name vectordb-milvus \
    --security-opt label=disable \
    -e ETCD_USE_EMBED=true \
    -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
    -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
    -e COMMON_STORAGETYPE=local \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/milvus/localdata:/var/lib/milvus \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/milvus/localdata/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml \
    -p 19530:19530 \
    -p 9091:9091 \
    -p 2379:2379 \
    --health-cmd="curl -f http://localhost:9091/healthz" \
    --health-interval=30s \
    --health-start-period=90s \
    --health-timeout=20s \
    --health-retries=3 \
    --network rag_network \
    milvusdb/milvus:v2.4.0 milvus run standalone \
#    1> /dev/null


## start streamlit locally
podman run -d --rm --security-opt label=disable \
    -p 8501:8501 \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/streamlit/localdata/:/opt/app-root/src/.cache/ \
    --network rag_network \
    quay.io/mdargatz/streamlitbot