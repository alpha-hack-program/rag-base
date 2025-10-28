# RAG Base Laboratory 🧪

A comprehensive Retrieval-Augmented Generation (RAG) laboratory for testing different combinations of Large Language Models (LLMs) and embedding models. This platform provides both a WebUI interface and OpenAI-compatible API access for experimentation and development.

## Architecture Overview

```
WebUI ──► Router ──► ┌─ Milvus (Vector Database)
                     ├─ Models (LLMs)  
                     └─ Embeddings (Vector Models)
```

The system uses a **router-based architecture** where:
- **WebUI**: User-friendly interface for interactive testing
- **Router**: Central component that manages model selection and request routing
- **Milvus**: Vector database for storing and retrieving document embeddings
- **Models**: Large Language Models (Granite, Llama, etc.)
- **Embeddings**: Vector embedding models for document processing

## 🚀 Quick Start

### Prerequisites

1. **ArgoCD** must be installed and running in your OpenShift cluster
2. **MinIO** must be deployed and accessible (required for document storage)
3. **GPU nodes** properly labeled for model deployment
4. **OpenShift/Kubernetes cluster** with appropriate resources

### Deployment Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/alpha-hack-program/rag-base.git
   cd rag-base
   ```

2. **Configure environment**
   ```bash
   cp env.sample .env
   # Edit .env with your specific configuration
   ```

3. **Deploy using ArgoCD**
   ```bash
   ./bootstrap/deploy.sh
   ```

## ⚙️ Configuration

### Environment Variables (.env)

The `.env` file is **mandatory** for deployment. Copy `env.sample` to `.env` and adapt the following key sections:

#### Required Configuration
- **ArgoCD Namespace**: Where ArgoCD is installed
- **Project Namespace**: Target namespace for RAG deployment
- **Git Repository**: Repository URL and branch
- **MinIO Credentials**: Access key, secret key, and endpoint

#### MinIO Configuration (MANDATORY)
MinIO is required for document storage and pipeline artifacts:
```env
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_ENDPOINT=minio.your-namespace.svc:9000
```

#### Node Selection Configuration
⚠️ **CRITICAL**: Configure node selection properly or label your nodes accordingly:

```env
NODE_SELECTOR_KEY=group
NODE_SELECTOR_VALUE=rag-base
GPU_TYPE=nvidia.com/gpu
```

**Label your nodes** with the configured selectors:
```bash
# Label nodes for general workloads
oc label node <node-name> group=rag-base

# Label GPU nodes for embedding models
oc label node <gpu-node-name> modelType=embedding

# Label GPU nodes for specific model types
oc label node <gpu-node-name> model=granite
oc label node <gpu-node-name> model=llama
```

The deployment script includes convenient labels for node selection:
- `modelType: 'embedding'` - For embedding model workloads
- `model: 'granite'` - For Granite LLM workloads  
- `model: 'llama'` - For Llama LLM workloads

**Either adapt these labels in the deployment configuration or label your nodes accordingly!**

## 🤖 Available Models

### Large Language Models (LLMs)
- **Granite 3.3 8B**: `granite-3-3-8b`
  - Model: `ibm-granite/granite-3.3-8b-instruct`
  - Features: Tool calling, enhanced instruction following
- **Llama 3.1 8B**: `llama-3-1-8b-w4a16`
  - Model: `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16`
  - Features: Quantized for efficiency, tool calling

### Embedding Models
- **Multilingual E5 Large**: `multilingual-e5-large-gpu`
  - Model: `intfloat/multilingual-e5-large`
  - Dimensions: 1024, Max tokens: 512
- **BGE M3**: `bge-m3-gpu`
  - Model: `BAAI/bge-m3`
  - Dimensions: 1024, Max tokens: 8192
- **Jina Embeddings v3**: `jina-embeddings-v3-gpu`
  - Model: `jinaai/jina-embeddings-v3`
  - Dimensions: 1024, Max tokens: 8192

## 🎯 Model Selection Strategy

**Important**: There is no default embeddings model. Instead, the system uses **composed model names** that allow you to select both the LLM and embedding model simultaneously through the router.

### Composed Model Names
The router exposes endpoints with combined model and embedding identifiers:
- Format: `{llm-model}+{embedding-model}`
- Example: `granite-3-3-8b+multilingual-e5-large-gpu`

This approach allows you to test different combinations:
- `granite-3-3-8b` with `bge-m3-gpu`
- `llama-3-1-8b-w4a16` with `jina-embeddings-v3-gpu`
- And any other combination

## 🌐 Access Methods

### 1. WebUI Interface
- User-friendly web interface for interactive testing
- Document upload and management
- Real-time model comparison
- Visual feedback and results

### 2. OpenAI-Compatible API
Access the RAG system programmatically using OpenAI SDK:

```python
from openai import OpenAI

# Configure client to use RAG router
client = OpenAI(
    base_url="http://router-endpoint/v1",
    api_key="not-needed"
)

# Use composed model names for selection
response = client.chat.completions.create(
    model="granite-3-3-8b+multilingual-e5-large-gpu",
    messages=[
        {"role": "user", "content": "Your question here"}
    ]
)
```

## 🗄️ Vector Database

**Milvus** is automatically deployed as the vector database for this RAG system:
- Stores document embeddings for retrieval
- Supports multiple collections for different embedding models
- Provides efficient similarity search
- Includes Attu web interface for database management

## 📋 Components Overview

### Core Services
- **Router**: Central API gateway with OpenAI-compatible interface
- **WebUI**: Interactive web application for testing
- **Milvus**: Vector database for embeddings storage
- **LSD (Llama Stack Distribution)**: Model serving infrastructure
- **MCP Servers**: Model Context Protocol servers for enhanced capabilities

### Data Pipeline
- **Kubeflow Pipelines**: Document processing and embedding generation
- **S3 Storage**: Document and artifact storage via MinIO
- **Vector Processing**: Automatic chunking and embedding generation

## 🔧 Troubleshooting

### Common Issues

1. **Node Selection Errors**
   - Ensure nodes are labeled with the configured selectors
   - Verify GPU availability and labels
   - Check node selector configuration in `.env`

2. **MinIO Connection Issues**
   - Verify MinIO credentials and endpoint
   - Ensure MinIO is accessible from the cluster
   - Check bucket creation and permissions

3. **ArgoCD Application Errors**
   - Confirm ArgoCD is running and accessible
   - Verify repository URL and credentials
   - Check namespace permissions

### Logs and Debugging
```bash
# Check ArgoCD application status
oc get applications -n openshift-gitops

# View router logs
oc logs -n <namespace> deployment/router

# Check model serving status
oc get inferenceservices -n <namespace>
```

## 🏗️ Development

### Adding New Models
1. Update `values.yaml` in `gitops/rag-base/`
2. Add model configuration with appropriate resource requirements
3. Configure node selectors and GPU requirements
4. Update router model mappings

### Customizing Prompts
Prompts are managed through ConfigMaps and can be customized:
- Edit prompt templates in router configuration
- Support for context-aware and context-free prompts
- Dynamic prompt injection for different use cases

## 📚 Example Documents

The repository includes sample documents in multiple languages:
- English: `/examples/documents/en/`
- Spanish: `/examples/documents/es/`
- Portuguese: `/examples/documents/pt/`
- LaTeX sources: `/examples/documents/latex/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with the full deployment
4. Submit a pull request

## 📄 License

This project is part of the Alpha Hack Program. See individual component licenses for details.

---

**⚠️ Important Notes:**
- Ensure proper GPU node labeling before deployment
- MinIO is a hard requirement - the system will not function without it
- The namespace configuration in `.env` is critical for proper component communication
- ArgoCD must be pre-installed and configured
- Test model combinations thoroughly to find optimal performance for your use cases