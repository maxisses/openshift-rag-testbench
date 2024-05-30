
## Install NVIDIA Driver

## Install Container Toolkit

## Run CDI Configuration Correctly
```sh
nvidia-ctk cdi list
```

## Provide Read Access to CDI Configuration
```sh
chmod + ... /etc/cdi/nvidia.yaml
```

## Create Common Network
```sh
podman network create rag_network
```

## Start VLLM Container Locally
```sh
podman run -d --rm --name vllm --security-opt label=disable --device nvidia.com/gpu=all \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/vllm/vllm-native/localdata/hub:/opt/app-root/src/.cache/huggingface/hub \
    -e HF_TOKEN=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 \
    -p 8000:8000 \
    --network rag_network \
    quay.io/rh-aiservices-bu/vllm-openai-ubi9:0.4.2 \
    --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ --dtype float16 --max-model-len 19872
```

## Start OLLAMA Container Locally
```sh
podman run -d --rm --name ollama --security-opt label=disable --device nvidia.com/gpu=all \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/ollama/localdata:/root/.ollama \
    -e HF_TOKEN=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 \
    -p 11434:11434 \
    --network rag_network \
    docker.io/ollama/ollama:latest
```

## Start Jupyter Locally
```sh
podman run -p 8888:8888 -d --rm \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/notebooks-for-ingestion/localdata:/opt/app-root/src/mydata \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/notebooks-for-ingestion/localdata/huggingface:/opt/app-root/src/.cache/ \
    --security-opt label=disable --device nvidia.com/gpu=all -e HF_TOKEN=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 \
    --network rag_network \
    quay.io/mdargatz/rag-ingest-workbench
```

## Start Milvus Locally
```sh
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
    milvusdb/milvus:v2.4.0 milvus run standalone
```

## Start Streamlit Locally
```sh
podman run -d --rm --security-opt label=disable \
    -p 8501:8501 \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/streamlit/localdata/:/opt/app-root/src/.cache/ \
    --network rag_network \
    quay.io/mdargatz/streamlitbot:latest
```

## Start Streamlit Locally in Dev mode
```sh
podman run --rm --security-opt label=disable \
    -p 8501:8501 \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/streamlit/localdata/:/opt/app-root/src/.cache/ \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/streamlit/rag-bot.py:/opt/app-root/src/rag-bot.py \
    -v /home/maxisses/TechStuff/openshift-rag-testbench/streamlit/.streamlit:/opt/app-root/src/.streamlit \
    --network rag_network \
    quay.io/mdargatz/streamlitbot:dev
```