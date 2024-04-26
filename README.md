## Infra SetUp

### a) MILVUS Vector DB
#### 1. ODF or AWS S3: Create a bucket
oc apply -f milvus/bucket-claim.yaml

#### 2. Download helm repo & Update milvus/openshift-values.yaml with your Object Bucket credentials or activate minio (which autogenerates for you but also spins up minio)

```bash
oc new-project <yourname>-chatbot
helm template -f openshift-values.yaml vectordb --set cluster.enabled=false --set etcd.replicaCount=1 --set pulsar.enabled=false milvus/milvus > milvus_manifest_standalone.yaml

yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-etcd") | .spec.template.spec.securityContext) = {}' -i milvus_manifest_standalone.yaml
yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-etcd") | .spec.template.spec.containers[0].securityContext) = {"capabilities": {"drop": ["ALL"]}, "runAsNonRoot": true, "allowPrivilegeEscalation": false}' -i milvus_manifest_standalone.yaml
yq '(select(.kind == "Deployment" and .metadata.name == "vectordb-minio") | .spec.template.spec.securityContext) = {"capabilities": {"drop": ["ALL"]}, "runAsNonRoot": true, "allowPrivilegeEscalation": false}' -i milvus_manifest_standalone.yaml
```

#### 3. apply to OpenShift
```bash
oc apply -f milvus/milvus_manifest_standalone.yaml
```
### b) Deploy ollama
```bash
oc apply -f ollama/
```

### C) Load Notebooks in OpenShift AI
#### 1. make NS available in RHOAI
```bash
oc patch namespace <yourname>-chatbot -p '{"metadata":{"labels":{"opendatahub.io/dashboard":"true"}}}' --type=merge
```
#### 2. Deploy a Workbench (e.g. medium, standard Data Science)
#### 3. clone the repo https://github.com/maxisses/openshift-rag-testbench 
#### 4. upload documents into the folders: docx, pptx, pdfs
#### 5. Run the ingest notebook - this take a while
#### 6. once done, run the ollama Notebook to test your RAG installation

### D) Deploy a Frontend
```bash
oc apply -f streamlit/k8s
oc create route edge --service=rag-frontend
```

### E) Alternative: Deploy vLLM via standard Deployment, Warning: GPU required, it loads mistral7B per default ~~requires approx 20GB RAM 
#### 1. Put a Secret.yaml in the vllm/vllm-native/, which contains your Huggingface token
```bash
oc apply -f vllm/vllm-native/
```




