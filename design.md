# Design Document: AI Model Watermarking System

## Overview

The AI Model Watermarking System provides a comprehensive framework for protecting intellectual property rights of deep learning models through digital watermarking. The system embeds cryptographically secure watermarks into trained model weights, enables ownership verification through correlation-based detection, and detects tampering attempts while preserving model performance.

The architecture integrates five core subsystems:

1. **Model Training Pipeline:** EfficientNet-B0 training on CIFAR-10 with AWOA-optimized hyperparameters
2. **Watermark Embedding Engine:** Binary watermark injection into classifier layer weights
3. **Verification System:** Cosine similarity-based watermark detection and validation
4. **Authentication Layer:** HMAC-SHA256 ownership certificate generation and verification
5. **Deployment Infrastructure:** Flask REST API and web-based user interface

The system achieves robust watermark detection (cosine similarity > 0.95) while maintaining model accuracy within 2% of baseline performance. It detects common attacks including weight pruning, noise injection, and fine-tuning through correlation threshold analysis.

## Architecture

### High-Level System Architecture

The system follows a layered architecture with clear separation between data processing, model operations, security, and presentation layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Web Frontend (HTML/CSS/JavaScript)                  │  │
│  │   - Training Interface                                │  │
│  │   - Watermark Embedding Interface                     │  │
│  │   - Verification Interface                            │  │
│  │   - Certificate Management Interface                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (Flask)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   REST Endpoints                                      │  │
│  │   /train  /embed  /detect  /verify  /certificate     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                      │
│  ┌──────────────┬──────────────┬──────────────────────┐    │
│  │   Training   │  Watermark   │   Authentication     │    │
│  │   Module     │  Module      │   Module             │    │
│  └──────────────┴──────────────┴──────────────────────┘    │
│  ┌──────────────┬──────────────┬──────────────────────┐    │
│  │   AWOA       │  Verification│   Tamper Detection   │    │
│  │   Optimizer  │  Module      │   Module             │    │
│  └──────────────┴──────────────┴──────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data/Storage Layer                        │
│  ┌──────────────┬──────────────┬──────────────────────┐    │
│  │  Model Store │  Watermark   │   Certificate        │    │
│  │  (.pth)      │  Store(.npy) │   Store (.pkl)       │    │
│  └──────────────┴──────────────┴──────────────────────┘    │
│  ┌──────────────┬──────────────────────────────────────┐   │
│  │  CIFAR-10    │  Model Registry (metadata.json)      │   │
│  │  Dataset     │                                       │   │
│  └──────────────┴──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```


### Design Principles

1. **Modularity:** Each component (training, embedding, verification) operates independently with well-defined interfaces
2. **Security by Design:** HMAC authentication and secure watermark storage prevent forgery
3. **Performance Preservation:** Watermark embedding minimizes accuracy degradation through alpha parameter tuning
4. **Robustness:** Multi-layered tamper detection identifies pruning, noise, and fine-tuning attacks
5. **Scalability:** Stateless API design supports concurrent operations
6. **Reproducibility:** Deterministic watermark generation with seed control

## Components and Interfaces

### 1. Dataset Module

**Purpose:** Load, preprocess, and augment CIFAR-10 dataset for model training

**Responsibilities:**
- Download CIFAR-10 dataset via torchvision
- Apply normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
- Resize images from 32x32 to 224x224 for EfficientNet-B0 input requirements
- Apply data augmentation (random horizontal flip, random crop with padding)
- Create PyTorch DataLoader instances with configurable batch size

**Interface:**
```python
class DatasetModule:
    def load_cifar10(train: bool) -> Dataset
    def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader
    def get_normalization_transform() -> transforms.Compose
    def get_augmentation_transform() -> transforms.Compose
```

**Inputs:**
- `train`: Boolean flag for training vs test split
- `batch_size`: Integer batch size for DataLoader
- `shuffle`: Boolean flag for data shuffling

**Outputs:**
- `Dataset`: PyTorch Dataset object containing CIFAR-10 images
- `DataLoader`: Iterator yielding batches of shape [batch_size, 3, 224, 224]

**Dependencies:**
- torchvision.datasets.CIFAR10
- torchvision.transforms
- torch.utils.data.DataLoader

### 2. Model Training Module

**Purpose:** Train EfficientNet-B0 on CIFAR-10 with optimized hyperparameters

**Responsibilities:**
- Initialize pretrained EfficientNet-B0 from torchvision
- Modify final classifier layer for 10-class CIFAR-10 output
- Execute training loop with Adam optimizer and CrossEntropyLoss
- Compute training and validation metrics per epoch
- Save trained model weights to .pth file
- Track training history (loss, accuracy curves)

**Interface:**
```python
class ModelTrainingModule:
    def initialize_model(num_classes: int, pretrained: bool) -> nn.Module
    def train_epoch(model: nn.Module, dataloader: DataLoader, 
                    optimizer: Optimizer, criterion: Loss) -> Tuple[float, float]
    def evaluate(model: nn.Module, dataloader: DataLoader, 
                 criterion: Loss) -> Tuple[float, float]
    def train_model(model: nn.Module, train_loader: DataLoader, 
                    val_loader: DataLoader, hyperparams: dict) -> nn.Module
    def save_model(model: nn.Module, path: str) -> None
    def load_model(path: str) -> nn.Module
```

**Inputs:**
- `num_classes`: Number of output classes (10 for CIFAR-10)
- `pretrained`: Boolean flag for using pretrained ImageNet weights
- `hyperparams`: Dictionary containing learning_rate, epochs, weight_decay, dropout

**Outputs:**
- `model`: Trained EfficientNet-B0 PyTorch model
- `train_loss`: Float training loss value
- `train_accuracy`: Float training accuracy percentage
- `val_loss`: Float validation loss value
- `val_accuracy`: Float validation accuracy percentage

**Dependencies:**
- torchvision.models.efficientnet_b0
- torch.optim.Adam
- torch.nn.CrossEntropyLoss

### 3. AWOA Optimization Module

**Purpose:** Optimize hyperparameters using Adaptive Whale Optimization Algorithm

**Responsibilities:**
- Initialize whale population with random hyperparameter values
- Evaluate fitness by training model and measuring validation accuracy
- Update whale positions using spiral mechanism and random search
- Adapt exploration/exploitation balance dynamically
- Return best hyperparameter configuration after convergence

**Interface:**
```python
class AWOAOptimizer:
    def __init__(self, search_space: dict, population_size: int, max_iterations: int)
    def initialize_population() -> List[dict]
    def evaluate_fitness(whale: dict) -> float
    def update_position(whale: dict, best_whale: dict, iteration: int) -> dict
    def optimize() -> dict
```

**Inputs:**
- `search_space`: Dictionary defining min/max ranges for each hyperparameter
  - `learning_rate`: [1e-5, 1e-2]
  - `batch_size`: [16, 128]
  - `weight_decay`: [1e-6, 1e-3]
  - `dropout`: [0.0, 0.5]
- `population_size`: Number of whale agents (typically 20-30)
- `max_iterations`: Maximum optimization iterations (typically 10-20)

**Outputs:**
- `best_hyperparams`: Dictionary containing optimized hyperparameter values
- `best_fitness`: Float validation accuracy achieved by best configuration
- `convergence_history`: List of best fitness values per iteration

**Algorithm Details:**
- Spiral updating equation: X(t+1) = D' * e^(bl) * cos(2πl) + X*(t)
- Random search: X(t+1) = X_rand - A * D
- Adaptive parameter a: linearly decreases from 2 to 0
- Coefficient A = 2a * r - a (r is random in [0,1])
- Coefficient C = 2 * r

**Dependencies:**
- numpy for numerical operations
- ModelTrainingModule for fitness evaluation

### 4. Watermark Embedding Module

**Purpose:** Embed binary digital watermark into classifier layer weights

**Responsibilities:**
- Generate random binary watermark matching classifier weight shape
- Embed watermark using additive formula: W' = W + α * M
- Validate embedding strength (alpha) through accuracy-robustness tradeoff
- Save watermarked model and original watermark separately
- Support deterministic watermark generation with seed control

**Interface:**
```python
class WatermarkEmbeddingModule:
    def generate_watermark(shape: Tuple[int, ...], seed: Optional[int]) -> np.ndarray
    def embed_watermark(model: nn.Module, watermark: np.ndarray, 
                        alpha: float) -> nn.Module
    def validate_alpha(model: nn.Module, watermark: np.ndarray, 
                       alpha_range: List[float], val_loader: DataLoader) -> dict
    def save_watermark(watermark: np.ndarray, path: str) -> None
    def load_watermark(path: str) -> np.ndarray
```

**Inputs:**
- `model`: Trained EfficientNet-B0 model
- `watermark`: Binary numpy array with shape matching classifier weights
- `alpha`: Embedding strength parameter (typically 0.01 to 0.5)
- `seed`: Optional random seed for reproducibility

**Outputs:**
- `watermarked_model`: Model with embedded watermark in classifier layer
- `watermark`: Original binary watermark array
- `validation_results`: Dictionary mapping alpha values to (accuracy, detection_confidence)

**Embedding Formula:**
```
W_watermarked = W_original + α * M
```
Where:
- W_original: Original classifier weights [num_classes, hidden_dim]
- M: Binary watermark {-1, +1}^(num_classes × hidden_dim)
- α: Embedding strength scalar
- W_watermarked: Modified classifier weights

**Dependencies:**
- numpy for array operations
- torch for model weight manipulation


### 5. Verification Module

**Purpose:** Detect and validate embedded watermarks using cosine similarity

**Responsibilities:**
- Extract watermark from suspected model weights
- Compute cosine similarity between original and extracted watermarks
- Compare similarity against detection threshold
- Report detection confidence and pass/fail status
- Support batch verification of multiple models

**Interface:**
```python
class VerificationModule:
    def extract_watermark(model: nn.Module, alpha: float) -> np.ndarray
    def compute_cosine_similarity(watermark1: np.ndarray, 
                                  watermark2: np.ndarray) -> float
    def verify_watermark(model: nn.Module, original_watermark: np.ndarray, 
                         alpha: float, threshold: float) -> dict
    def batch_verify(models: List[nn.Module], watermark: np.ndarray, 
                     alpha: float, threshold: float) -> List[dict]
```

**Inputs:**
- `model`: Suspected watermarked model
- `original_watermark`: Original binary watermark array
- `alpha`: Embedding strength used during embedding
- `threshold`: Minimum cosine similarity for successful detection (typically 0.90-0.95)

**Outputs:**
- `extracted_watermark`: Watermark extracted from model weights
- `cosine_similarity`: Float similarity score in [-1, 1]
- `detection_status`: Boolean indicating pass (similarity > threshold) or fail
- `confidence`: Normalized confidence score in [0, 1]

**Extraction Formula:**
```
M_extracted = (W_watermarked - W_original) / α
```

**Cosine Similarity:**
```
similarity = (M_original · M_extracted) / (||M_original|| * ||M_extracted||)
```

**Dependencies:**
- numpy for array operations
- scipy.spatial.distance.cosine (alternative implementation)

### 6. Authentication Module

**Purpose:** Generate and verify HMAC-SHA256 ownership certificates

**Responsibilities:**
- Create ownership certificate with owner metadata and watermark hash
- Compute HMAC-SHA256 signature using secret key
- Serialize certificate to .pkl file
- Verify certificate authenticity by recomputing HMAC
- Link certificates with license information

**Interface:**
```python
class AuthenticationModule:
    def __init__(self, secret_key: bytes)
    def generate_certificate(owner_name: str, model_id: str, 
                             watermark_hash: str, license_info: dict) -> dict
    def compute_hmac(certificate_data: dict) -> str
    def verify_certificate(certificate: dict) -> bool
    def save_certificate(certificate: dict, path: str) -> None
    def load_certificate(path: str) -> dict
    def link_license(certificate: dict, license_record: dict) -> dict
```

**Inputs:**
- `secret_key`: 256-bit cryptographic key for HMAC computation
- `owner_name`: String identifying model owner
- `model_id`: Unique identifier for watermarked model
- `watermark_hash`: SHA-256 hash of original watermark
- `license_info`: Dictionary containing license type, expiration, restrictions

**Outputs:**
- `certificate`: Dictionary containing:
  - `owner_name`: String
  - `model_id`: String
  - `watermark_hash`: String (SHA-256 hex digest)
  - `timestamp`: ISO 8601 timestamp
  - `license_info`: Dictionary
  - `hmac_signature`: String (HMAC-SHA256 hex digest)
- `verification_result`: Boolean indicating valid/invalid signature

**Certificate Structure:**
```python
{
    "owner_name": "John Doe",
    "model_id": "efficientnet_b0_cifar10_001",
    "watermark_hash": "a3f5...",
    "timestamp": "2024-01-15T10:30:00Z",
    "license_info": {
        "type": "commercial",
        "licensee": "Company XYZ",
        "expiration": "2025-01-15",
        "restrictions": ["no_redistribution", "single_deployment"]
    },
    "hmac_signature": "b7e2..."
}
```

**HMAC Computation:**
```
signature = HMAC-SHA256(secret_key, 
                        owner_name || model_id || watermark_hash || timestamp)
```

**Dependencies:**
- hashlib for SHA-256 and HMAC-SHA256
- pickle for certificate serialization
- datetime for timestamp generation

### 7. Tamper Detection Module

**Purpose:** Detect unauthorized modifications to watermarked models

**Responsibilities:**
- Detect weight pruning by identifying zeroed or removed weights
- Detect noise injection through weight distribution analysis
- Detect fine-tuning by measuring weight deviation from original
- Compute residual cosine similarity after suspected tampering
- Report attack type and confidence level

**Interface:**
```python
class TamperDetectionModule:
    def detect_pruning(model: nn.Module, original_model: nn.Module) -> dict
    def detect_noise_injection(model: nn.Module, original_model: nn.Module) -> dict
    def detect_finetuning(model: nn.Module, original_model: nn.Module, 
                          threshold: float) -> dict
    def analyze_tampering(model: nn.Module, original_model: nn.Module, 
                          watermark: np.ndarray, alpha: float) -> dict
```

**Inputs:**
- `model`: Suspected tampered model
- `original_model`: Original watermarked model (reference)
- `watermark`: Original binary watermark
- `alpha`: Embedding strength
- `threshold`: Maximum acceptable weight deviation

**Outputs:**
- `attack_detected`: Boolean indicating tampering detection
- `attack_type`: String in ["pruning", "noise_injection", "fine_tuning", "none"]
- `confidence`: Float confidence score in [0, 1]
- `residual_similarity`: Cosine similarity after tampering
- `weight_deviation`: L2 norm of weight differences

**Detection Methods:**

1. **Pruning Detection:**
   - Count zero weights in classifier layer
   - Compare sparsity with original model
   - Flag if sparsity increase > 10%

2. **Noise Injection Detection:**
   - Compute weight distribution statistics (mean, std, kurtosis)
   - Compare with original distribution using KL divergence
   - Flag if KL divergence > threshold

3. **Fine-tuning Detection:**
   - Compute L2 norm: ||W_suspected - W_original||
   - Normalize by total weight magnitude
   - Flag if normalized deviation > threshold (typically 0.1)

**Dependencies:**
- numpy for statistical analysis
- scipy.stats for distribution comparison
- VerificationModule for residual similarity computation

### 8. Deployment Module

**Purpose:** Manage model persistence, registry, and API integration

**Responsibilities:**
- Save/load models, watermarks, and certificates with consistent naming
- Maintain model registry mapping identifiers to file paths
- Verify file integrity using checksums
- Provide unified interface for storage operations
- Support batch operations for multiple models

**Interface:**
```python
class DeploymentModule:
    def __init__(self, storage_root: str)
    def save_watermarked_model(model: nn.Module, watermark: np.ndarray, 
                               certificate: dict, model_id: str) -> dict
    def load_watermarked_model(model_id: str) -> Tuple[nn.Module, np.ndarray, dict]
    def register_model(model_id: str, metadata: dict) -> None
    def get_model_metadata(model_id: str) -> dict
    def list_models() -> List[str]
    def verify_file_integrity(file_path: str, expected_checksum: str) -> bool
```

**Inputs:**
- `storage_root`: Base directory for all stored files
- `model_id`: Unique identifier for model (e.g., "efficientnet_b0_001")
- `model`: PyTorch model to save
- `watermark`: NumPy watermark array
- `certificate`: Ownership certificate dictionary

**Outputs:**
- `file_paths`: Dictionary mapping file types to paths
  - `model_path`: Path to .pth file
  - `watermark_path`: Path to .npy file
  - `certificate_path`: Path to .pkl file
- `metadata`: Dictionary containing model information, timestamps, checksums

**File Naming Convention:**
```
{storage_root}/
  models/
    {model_id}.pth
  watermarks/
    {model_id}.npy
  certificates/
    {model_id}.pkl
  registry.json
```

**Registry Structure:**
```json
{
  "efficientnet_b0_001": {
    "model_path": "models/efficientnet_b0_001.pth",
    "watermark_path": "watermarks/efficientnet_b0_001.npy",
    "certificate_path": "certificates/efficientnet_b0_001.pkl",
    "created_at": "2024-01-15T10:30:00Z",
    "owner": "John Doe",
    "checksum_model": "sha256:a3f5...",
    "checksum_watermark": "sha256:b7e2...",
    "alpha": 0.1,
    "accuracy": 0.89
  }
}
```

**Dependencies:**
- torch for model serialization
- numpy for watermark serialization
- pickle for certificate serialization
- hashlib for checksum computation
- json for registry management

## Data Models

### Model Weights Structure

**EfficientNet-B0 Architecture:**
```
EfficientNet-B0
├── features (convolutional blocks)
│   ├── Conv2d (3 → 32)
│   ├── MBConv blocks (32 → 1280)
│   └── Conv2d (1280 → 1280)
├── avgpool (AdaptiveAvgPool2d)
└── classifier (Linear: 1280 → 10)  ← Watermark embedded here
```

**Classifier Layer Shape:**
- Input dimension: 1280 (EfficientNet-B0 feature dimension)
- Output dimension: 10 (CIFAR-10 classes)
- Weight tensor shape: [10, 1280]
- Bias tensor shape: [10]
- Watermark shape: [10, 1280] (matches weight shape)

### Watermark Data Model

**Binary Watermark:**
```python
watermark = {
    "data": np.ndarray,        # Shape: [10, 1280], dtype: float32, values: {-1, +1}
    "shape": (10, 1280),       # Tuple
    "seed": 42,                # Optional int for reproducibility
    "hash": "sha256:a3f5...",  # SHA-256 hash for integrity
    "created_at": "2024-01-15T10:30:00Z"
}
```

### AWOA Whale Agent

**Hyperparameter Configuration:**
```python
whale = {
    "learning_rate": 0.001,    # Float in [1e-5, 1e-2]
    "batch_size": 64,          # Int in [16, 128]
    "weight_decay": 1e-4,      # Float in [1e-6, 1e-3]
    "dropout": 0.2,            # Float in [0.0, 0.5]
    "fitness": 0.89,           # Validation accuracy
    "position": [0.3, 0.5, 0.2, 0.4]  # Normalized position in search space
}
```

### License Record

**License Information:**
```python
license_record = {
    "license_id": "LIC-2024-001",
    "model_id": "efficientnet_b0_001",
    "licensee_name": "Company XYZ",
    "license_type": "commercial",  # ["commercial", "academic", "evaluation"]
    "issued_at": "2024-01-15T10:30:00Z",
    "expires_at": "2025-01-15T10:30:00Z",
    "restrictions": ["no_redistribution", "single_deployment"],
    "status": "active"  # ["active", "expired", "revoked"]
}
```

### Verification Result

**Detection Output:**
```python
verification_result = {
    "model_id": "efficientnet_b0_001",
    "cosine_similarity": 0.97,
    "threshold": 0.95,
    "detection_status": "pass",  # ["pass", "fail"]
    "confidence": 0.97,
    "timestamp": "2024-01-15T10:30:00Z",
    "tamper_detected": False,
    "attack_type": "none"  # ["none", "pruning", "noise_injection", "fine_tuning"]
}
```

## Data Flow Design

### End-to-End Watermarking Flow

```
1. Dataset Preparation
   CIFAR-10 → Preprocessing → DataLoader
   
2. Hyperparameter Optimization
   AWOA → Fitness Evaluation → Best Hyperparameters
   
3. Model Training
   EfficientNet-B0 + Hyperparameters → Training Loop → Trained Model
   
4. Watermark Generation
   Random Seed → Binary Watermark [10, 1280]
   
5. Watermark Embedding
   Trained Model + Watermark + Alpha → Watermarked Model
   
6. Alpha Validation
   Watermarked Model → Accuracy Test + Detection Test → Optimal Alpha
   
7. Certificate Generation
   Owner Info + Watermark Hash → HMAC Signature → Certificate
   
8. Model Deployment
   Watermarked Model + Watermark + Certificate → Storage → Registry
```

### Verification Flow

```
1. Model Loading
   Model ID → Load Model + Watermark + Certificate
   
2. Watermark Extraction
   Model Weights → (W - W_original) / Alpha → Extracted Watermark
   
3. Similarity Computation
   Original Watermark · Extracted Watermark → Cosine Similarity
   
4. Threshold Comparison
   Cosine Similarity vs Threshold → Pass/Fail Decision
   
5. Tamper Detection
   Model vs Original → Pruning/Noise/Fine-tuning Analysis
   
6. Certificate Verification
   Certificate → HMAC Recomputation → Valid/Invalid
   
7. Result Reporting
   Detection Status + Confidence + Tamper Analysis → Verification Report
```

### API Request Flow

```
Client Request → Flask Route → Input Validation → Business Logic → Storage/Computation → Response Formation → Client Response
```

**Example: Watermark Embedding Request**
```
POST /embed
{
  "model_id": "efficientnet_b0_001",
  "alpha": 0.1,
  "seed": 42
}
↓
Load trained model from storage
↓
Generate watermark with seed
↓
Embed watermark in classifier weights
↓
Save watermarked model, watermark, metadata
↓
{
  "status": "success",
  "watermarked_model_id": "efficientnet_b0_001_watermarked",
  "cosine_similarity": 0.98,
  "accuracy_degradation": 0.015
}
```


## Algorithm Design

### EfficientNet-B0 Training Pipeline

**Algorithm: Train EfficientNet-B0 on CIFAR-10**

```
Input: train_loader, val_loader, hyperparameters
Output: trained_model, training_history

1. Initialize model = EfficientNet-B0(pretrained=True)
2. Modify classifier: model.classifier = Linear(1280, 10)
3. Initialize optimizer = Adam(model.parameters(), lr=hyperparameters.learning_rate, 
                               weight_decay=hyperparameters.weight_decay)
4. Initialize criterion = CrossEntropyLoss()
5. Move model to GPU if available

6. For epoch in range(hyperparameters.epochs):
   a. Set model to training mode
   b. For batch in train_loader:
      i.   images, labels = batch
      ii.  outputs = model(images)
      iii. loss = criterion(outputs, labels)
      iv.  optimizer.zero_grad()
      v.   loss.backward()
      vi.  optimizer.step()
   
   c. Set model to evaluation mode
   d. For batch in val_loader:
      i.   images, labels = batch
      ii.  outputs = model(images)
      iii. Compute accuracy
   
   e. Record epoch metrics (train_loss, train_acc, val_loss, val_acc)
   f. If val_acc > best_val_acc:
      i. best_val_acc = val_acc
      ii. Save model checkpoint

7. Return trained_model, training_history
```

**Time Complexity:** O(epochs × batches × forward_pass)
- Forward pass: O(n × m) where n = batch_size, m = model_parameters
- Typical training time: 30-60 minutes on GPU for 50 epochs

### AWOA Optimization Algorithm

**Algorithm: Adaptive Whale Optimization for Hyperparameter Tuning**

```
Input: search_space, population_size, max_iterations
Output: best_hyperparameters, best_fitness

1. Initialize population of N whales with random positions in search_space
2. Evaluate fitness for each whale (train model, measure val_accuracy)
3. Identify best_whale with highest fitness
4. Set a = 2 (exploration parameter)

5. For iteration in range(max_iterations):
   a. Update a = 2 - iteration × (2 / max_iterations)  # Linear decrease
   
   b. For each whale in population:
      i.   Generate random values r, l in [0, 1]
      ii.  Compute A = 2 × a × r - a
      iii. Compute C = 2 × r
      
      iv.  If |A| < 1:  # Exploitation phase
           - Compute D = |C × best_whale.position - whale.position|
           - If random() < 0.5:  # Encircling prey
               whale.position = best_whale.position - A × D
           - Else:  # Spiral updating
               D' = |best_whale.position - whale.position|
               whale.position = D' × exp(b×l) × cos(2πl) + best_whale.position
      
      v.   Else:  # Exploration phase
           - Select random_whale from population
           - Compute D = |C × random_whale.position - whale.position|
           - whale.position = random_whale.position - A × D
      
      vi.  Clip whale.position to search_space bounds
      vii. Decode position to hyperparameters
      viii. Evaluate fitness (train model, measure val_accuracy)
      ix.  If whale.fitness > best_whale.fitness:
               best_whale = whale
   
   c. Record convergence history

6. Return best_whale.hyperparameters, best_whale.fitness
```

**Parameters:**
- b = 1 (spiral shape constant)
- population_size = 20-30
- max_iterations = 10-20

**Time Complexity:** O(iterations × population_size × training_time)
- Typical optimization time: 4-8 hours on GPU

### Watermark Embedding Algorithm

**Algorithm: Embed Binary Watermark in Classifier Weights**

```
Input: model, watermark, alpha
Output: watermarked_model

1. Extract classifier layer: classifier = model.classifier
2. Get original weights: W_original = classifier.weight.data  # Shape: [10, 1280]
3. Verify watermark shape matches W_original.shape
4. Convert watermark to torch tensor on same device as model
5. Compute watermarked weights: W_watermarked = W_original + alpha × watermark
6. Update classifier weights: classifier.weight.data = W_watermarked
7. Return model (modified in-place)
```

**Mathematical Formula:**
```
W'[i,j] = W[i,j] + α × M[i,j]

Where:
- W[i,j]: Original weight at position (i,j)
- M[i,j]: Binary watermark value ∈ {-1, +1}
- α: Embedding strength (typically 0.01 to 0.5)
- W'[i,j]: Watermarked weight
```

**Time Complexity:** O(n × m) where n=10, m=1280
- Embedding time: < 1 second

### Watermark Detection Algorithm

**Algorithm: Extract and Verify Watermark**

```
Input: watermarked_model, original_watermark, alpha, threshold
Output: verification_result

1. Extract classifier weights: W_watermarked = watermarked_model.classifier.weight.data
2. Load original model weights: W_original (if available)
3. Extract watermark: M_extracted = (W_watermarked - W_original) / alpha
4. Convert to numpy arrays: M_extracted_np, M_original_np
5. Flatten arrays: M_extracted_flat, M_original_flat
6. Compute cosine similarity:
   a. dot_product = M_original_flat · M_extracted_flat
   b. norm_original = ||M_original_flat||
   c. norm_extracted = ||M_extracted_flat||
   d. cosine_similarity = dot_product / (norm_original × norm_extracted)
7. Compare with threshold:
   a. If cosine_similarity >= threshold: detection_status = "pass"
   b. Else: detection_status = "fail"
8. Return {cosine_similarity, detection_status, confidence}
```

**Cosine Similarity Formula:**
```
similarity = Σ(M_original[i] × M_extracted[i]) / (√Σ(M_original[i]²) × √Σ(M_extracted[i]²))

Range: [-1, 1]
- 1.0: Perfect match
- 0.0: Orthogonal (no correlation)
- -1.0: Perfect opposite
```

**Time Complexity:** O(n × m) where n=10, m=1280
- Detection time: < 1 second

### Alpha Validation Algorithm

**Algorithm: Find Optimal Embedding Strength**

```
Input: model, watermark, alpha_range, val_loader, threshold
Output: optimal_alpha, validation_results

1. Initialize results = []
2. For alpha in alpha_range:
   a. Create copy of model: model_copy = deepcopy(model)
   b. Embed watermark: watermarked_model = embed_watermark(model_copy, watermark, alpha)
   c. Evaluate accuracy: val_accuracy = evaluate(watermarked_model, val_loader)
   d. Detect watermark: detection_result = verify_watermark(watermarked_model, watermark, alpha)
   e. Compute accuracy_degradation = original_accuracy - val_accuracy
   f. Record: results.append({
        "alpha": alpha,
        "accuracy": val_accuracy,
        "degradation": accuracy_degradation,
        "cosine_similarity": detection_result.cosine_similarity,
        "detection_status": detection_result.detection_status
      })

3. Filter results where detection_status == "pass"
4. Sort by minimum accuracy_degradation
5. Select optimal_alpha with degradation < 2% and highest cosine_similarity
6. Return optimal_alpha, results
```

**Typical Alpha Range:** [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

**Selection Criteria:**
- Accuracy degradation < 2%
- Cosine similarity > 0.95
- Prefer higher alpha for robustness if both criteria met

### HMAC Certificate Generation Algorithm

**Algorithm: Generate Authenticated Ownership Certificate**

```
Input: owner_name, model_id, watermark, secret_key, license_info
Output: certificate

1. Compute watermark_hash = SHA256(watermark.tobytes())
2. Get current timestamp = datetime.now().isoformat()
3. Create certificate_data = {
     "owner_name": owner_name,
     "model_id": model_id,
     "watermark_hash": watermark_hash,
     "timestamp": timestamp,
     "license_info": license_info
   }
4. Serialize data: message = owner_name || model_id || watermark_hash || timestamp
5. Compute HMAC: hmac_signature = HMAC-SHA256(secret_key, message)
6. Add signature to certificate: certificate_data["hmac_signature"] = hmac_signature
7. Return certificate_data
```

**HMAC-SHA256 Computation:**
```
HMAC(K, m) = H((K ⊕ opad) || H((K ⊕ ipad) || m))

Where:
- K: Secret key (256 bits)
- m: Message (concatenated certificate fields)
- H: SHA-256 hash function
- opad: Outer padding (0x5c repeated)
- ipad: Inner padding (0x36 repeated)
- ||: Concatenation
- ⊕: XOR operation
```

**Time Complexity:** O(n) where n = message length
- Certificate generation time: < 10 milliseconds

### Tamper Detection Algorithm

**Algorithm: Detect Model Tampering**

```
Input: suspected_model, original_model, watermark, alpha
Output: tamper_report

1. Initialize tamper_report = {
     "pruning_detected": False,
     "noise_detected": False,
     "finetuning_detected": False,
     "residual_similarity": 0.0
   }

2. Extract weights:
   W_suspected = suspected_model.classifier.weight.data
   W_original = original_model.classifier.weight.data

3. Pruning Detection:
   a. Count zeros: zero_count_suspected = count_zeros(W_suspected)
   b. Count zeros: zero_count_original = count_zeros(W_original)
   c. Compute sparsity_increase = (zero_count_suspected - zero_count_original) / total_weights
   d. If sparsity_increase > 0.1:
      tamper_report["pruning_detected"] = True

4. Noise Injection Detection:
   a. Compute statistics: mean_s, std_s = compute_stats(W_suspected)
   b. Compute statistics: mean_o, std_o = compute_stats(W_original)
   c. Compute KL divergence: kl_div = KL(P_suspected || P_original)
   d. If kl_div > threshold_kl:
      tamper_report["noise_detected"] = True

5. Fine-tuning Detection:
   a. Compute weight difference: diff = W_suspected - W_original
   b. Compute L2 norm: l2_norm = ||diff||
   c. Normalize: normalized_deviation = l2_norm / ||W_original||
   d. If normalized_deviation > 0.1:
      tamper_report["finetuning_detected"] = True

6. Residual Watermark Detection:
   a. Extract watermark: M_extracted = (W_suspected - W_original) / alpha
   b. Compute similarity: residual_similarity = cosine_similarity(watermark, M_extracted)
   c. tamper_report["residual_similarity"] = residual_similarity

7. Return tamper_report
```

**Detection Thresholds:**
- Pruning: sparsity_increase > 10%
- Noise: KL divergence > 0.5
- Fine-tuning: normalized_deviation > 10%
- Residual similarity < 0.90 indicates significant tampering

## Database / Storage Design

### File System Structure

```
storage_root/
├── models/
│   ├── efficientnet_b0_001.pth
│   ├── efficientnet_b0_002.pth
│   └── ...
├── watermarks/
│   ├── efficientnet_b0_001.npy
│   ├── efficientnet_b0_002.npy
│   └── ...
├── certificates/
│   ├── efficientnet_b0_001.pkl
│   ├── efficientnet_b0_002.pkl
│   └── ...
├── datasets/
│   └── cifar-10-batches-py/
├── registry.json
└── config.json
```

### Model Storage (.pth)

**Format:** PyTorch state dictionary
**Content:**
```python
{
  "model_state_dict": OrderedDict([
    ("features.0.weight", Tensor[...]),
    ("classifier.weight", Tensor[10, 1280]),  # Watermarked weights
    ("classifier.bias", Tensor[10]),
    ...
  ]),
  "metadata": {
    "architecture": "efficientnet_b0",
    "num_classes": 10,
    "input_size": [3, 224, 224],
    "watermarked": True,
    "alpha": 0.1
  }
}
```

**Size:** ~20 MB per model

### Watermark Storage (.npy)

**Format:** NumPy binary array
**Content:**
```python
watermark = np.array([...], dtype=np.float32)  # Shape: [10, 1280]
```

**Metadata (separate JSON):**
```json
{
  "shape": [10, 1280],
  "dtype": "float32",
  "seed": 42,
  "hash": "sha256:a3f5...",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Size:** ~50 KB per watermark

### Certificate Storage (.pkl)

**Format:** Python pickle
**Content:**
```python
{
  "owner_name": "John Doe",
  "model_id": "efficientnet_b0_001",
  "watermark_hash": "sha256:a3f5...",
  "timestamp": "2024-01-15T10:30:00Z",
  "license_info": {...},
  "hmac_signature": "b7e2..."
}
```

**Size:** ~1 KB per certificate

### Registry (registry.json)

**Format:** JSON
**Content:**
```json
{
  "models": {
    "efficientnet_b0_001": {
      "model_path": "models/efficientnet_b0_001.pth",
      "watermark_path": "watermarks/efficientnet_b0_001.npy",
      "certificate_path": "certificates/efficientnet_b0_001.pkl",
      "created_at": "2024-01-15T10:30:00Z",
      "owner": "John Doe",
      "alpha": 0.1,
      "accuracy": 0.89,
      "checksum_model": "sha256:...",
      "checksum_watermark": "sha256:...",
      "status": "active"
    }
  },
  "metadata": {
    "version": "1.0",
    "last_updated": "2024-01-15T10:30:00Z"
  }
}
```

### Configuration (config.json)

**Format:** JSON
**Content:**
```json
{
  "storage_root": "/path/to/storage",
  "secret_key_path": "/secure/path/to/secret.key",
  "detection_threshold": 0.95,
  "default_alpha": 0.1,
  "alpha_range": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
  "awoa_config": {
    "population_size": 25,
    "max_iterations": 15,
    "search_space": {
      "learning_rate": [1e-5, 1e-2],
      "batch_size": [16, 128],
      "weight_decay": [1e-6, 1e-3],
      "dropout": [0.0, 0.5]
    }
  },
  "training_config": {
    "epochs": 50,
    "device": "cuda",
    "num_workers": 4
  }
}
```


## API Design

### Flask REST API Endpoints

**Base URL:** `http://localhost:5000/api/v1`

### 1. Train Model Endpoint

**POST /train**

Train EfficientNet-B0 on CIFAR-10 with AWOA-optimized hyperparameters.

**Request:**
```json
{
  "use_awoa": true,
  "awoa_iterations": 15,
  "manual_hyperparams": {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "weight_decay": 1e-4
  }
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "model_id": "efficientnet_b0_001",
  "accuracy": 0.89,
  "loss": 0.32,
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "weight_decay": 1e-4
  },
  "training_time": 3600
}
```

**Response (Error - 400):**
```json
{
  "status": "error",
  "message": "Invalid hyperparameter range",
  "details": "learning_rate must be between 1e-5 and 1e-2"
}
```

### 2. Embed Watermark Endpoint

**POST /embed**

Embed digital watermark into trained model.

**Request:**
```json
{
  "model_id": "efficientnet_b0_001",
  "alpha": 0.1,
  "seed": 42,
  "owner_name": "John Doe",
  "license_info": {
    "type": "commercial",
    "licensee": "Company XYZ",
    "expiration": "2025-01-15"
  }
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "watermarked_model_id": "efficientnet_b0_001_watermarked",
  "watermark_id": "watermark_001",
  "certificate_id": "cert_001",
  "cosine_similarity": 0.98,
  "accuracy_before": 0.89,
  "accuracy_after": 0.88,
  "accuracy_degradation": 0.01
}
```

### 3. Verify Watermark Endpoint

**POST /verify**

Detect and verify watermark in suspected model.

**Request:**
```json
{
  "model_id": "efficientnet_b0_001_watermarked",
  "watermark_id": "watermark_001",
  "alpha": 0.1,
  "threshold": 0.95
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "detection_status": "pass",
  "cosine_similarity": 0.97,
  "threshold": 0.95,
  "confidence": 0.97,
  "tamper_detected": false,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 4. Validate Alpha Endpoint

**POST /validate-alpha**

Test multiple alpha values to find optimal embedding strength.

**Request:**
```json
{
  "model_id": "efficientnet_b0_001",
  "watermark_id": "watermark_001",
  "alpha_range": [0.01, 0.05, 0.1, 0.15, 0.2]
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "optimal_alpha": 0.1,
  "results": [
    {
      "alpha": 0.01,
      "accuracy": 0.89,
      "degradation": 0.0,
      "cosine_similarity": 0.92,
      "detection_status": "fail"
    },
    {
      "alpha": 0.1,
      "accuracy": 0.88,
      "degradation": 0.01,
      "cosine_similarity": 0.98,
      "detection_status": "pass"
    }
  ]
}
```

### 5. Verify Certificate Endpoint

**POST /verify-certificate**

Verify HMAC signature of ownership certificate.

**Request:**
```json
{
  "certificate_id": "cert_001"
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "valid": true,
  "owner_name": "John Doe",
  "model_id": "efficientnet_b0_001_watermarked",
  "timestamp": "2024-01-15T10:30:00Z",
  "license_info": {
    "type": "commercial",
    "licensee": "Company XYZ",
    "status": "active"
  }
}
```

### 6. Detect Tampering Endpoint

**POST /detect-tampering**

Analyze model for pruning, noise injection, or fine-tuning attacks.

**Request:**
```json
{
  "suspected_model_id": "efficientnet_b0_001_modified",
  "original_model_id": "efficientnet_b0_001_watermarked",
  "watermark_id": "watermark_001",
  "alpha": 0.1
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "tampering_detected": true,
  "attack_types": ["fine_tuning"],
  "pruning_detected": false,
  "noise_detected": false,
  "finetuning_detected": true,
  "residual_similarity": 0.85,
  "weight_deviation": 0.12,
  "confidence": 0.88
}
```

### 7. Predict Endpoint

**POST /predict**

Run inference on watermarked model.

**Request:**
```json
{
  "model_id": "efficientnet_b0_001_watermarked",
  "image": "base64_encoded_image_data"
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "predictions": [
    {"class": "airplane", "probability": 0.92},
    {"class": "bird", "probability": 0.05},
    {"class": "ship", "probability": 0.02}
  ],
  "inference_time": 0.015
}
```

### 8. List Models Endpoint

**GET /models**

Retrieve list of all registered models.

**Response (Success - 200):**
```json
{
  "status": "success",
  "models": [
    {
      "model_id": "efficientnet_b0_001",
      "owner": "John Doe",
      "created_at": "2024-01-15T10:30:00Z",
      "watermarked": true,
      "accuracy": 0.89,
      "status": "active"
    }
  ],
  "total": 1
}
```

### Error Handling

**Standard Error Response:**
```json
{
  "status": "error",
  "error_code": "INVALID_MODEL_ID",
  "message": "Model not found",
  "details": "No model with ID 'efficientnet_b0_999' exists in registry"
}
```

**HTTP Status Codes:**
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (model/watermark/certificate not found)
- 500: Internal Server Error (computation failure)

### CORS Configuration

```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

## Security Design

### HMAC-SHA256 Authentication

**Purpose:** Cryptographically authenticate ownership certificates to prevent forgery.

**Implementation:**

1. **Secret Key Management:**
   - Generate 256-bit random key using `secrets.token_bytes(32)`
   - Store in secure location with restricted file permissions (chmod 600)
   - Never expose key in API responses or logs
   - Rotate keys periodically (recommended: every 6 months)

2. **Certificate Signing:**
   ```python
   import hmac
   import hashlib
   
   def sign_certificate(certificate_data, secret_key):
       message = f"{certificate_data['owner_name']}|{certificate_data['model_id']}|{certificate_data['watermark_hash']}|{certificate_data['timestamp']}"
       signature = hmac.new(
           secret_key.encode(),
           message.encode(),
           hashlib.sha256
       ).hexdigest()
       return signature
   ```

3. **Certificate Verification:**
   ```python
   def verify_certificate(certificate, secret_key):
       stored_signature = certificate['hmac_signature']
       computed_signature = sign_certificate(certificate, secret_key)
       return hmac.compare_digest(stored_signature, computed_signature)
   ```

**Security Properties:**
- Collision resistance: SHA-256 provides 128-bit security
- Unforgeability: Cannot create valid signature without secret key
- Timing attack resistance: `hmac.compare_digest` prevents timing attacks

### Watermark Confidentiality

**Threat Model:**
- Attacker has access to watermarked model
- Attacker does NOT have access to original watermark array
- Attacker may attempt to remove or forge watermarks

**Protection Mechanisms:**

1. **Watermark Storage:**
   - Store original watermark separately from model
   - Restrict file permissions (chmod 600)
   - Never transmit watermark over network without encryption
   - Use secure channels (HTTPS) for API communication

2. **Detection Security:**
   - Verification requires both watermarked model AND original watermark
   - Without original watermark, attacker cannot verify detection threshold
   - Cosine similarity computation requires exact watermark values

3. **Embedding Strength:**
   - Alpha parameter balances detectability and security
   - Higher alpha: easier to detect, but more visible to attackers
   - Lower alpha: harder to detect, but more robust against removal
   - Recommended: alpha = 0.1 (good balance)

### Correlation Thresholding

**Purpose:** Distinguish legitimate watermarks from random noise.

**Threshold Selection:**
- Default threshold: 0.95
- Rationale: Cosine similarity > 0.95 indicates strong correlation
- False positive rate: < 0.01% (empirically validated)
- False negative rate: < 1% for unmodified models

**Statistical Analysis:**
```
H0 (null hypothesis): Model does not contain watermark
H1 (alternative): Model contains watermark

Test statistic: cosine_similarity(M_original, M_extracted)

Decision rule:
- If cosine_similarity >= 0.95: Reject H0 (watermark detected)
- If cosine_similarity < 0.95: Fail to reject H0 (no watermark)
```

**Threshold Calibration:**
1. Generate 1000 random watermarks
2. Compute pairwise cosine similarities
3. Measure distribution (mean ≈ 0, std ≈ 0.03)
4. Set threshold at mean + 3×std ≈ 0.09 (for random noise)
5. Legitimate watermarks achieve similarity > 0.95 (well above noise)

### Attack Resistance

**1. Watermark Removal Attacks:**
- **Fine-tuning:** Detected by weight deviation analysis (L2 norm)
- **Pruning:** Detected by sparsity increase analysis
- **Noise injection:** Detected by distribution analysis (KL divergence)
- **Overwriting:** Detected by cosine similarity drop below threshold

**2. Forgery Attacks:**
- **Certificate forgery:** Prevented by HMAC-SHA256 signature
- **Watermark forgery:** Requires knowledge of original watermark (kept secret)
- **Model substitution:** Detected by watermark verification failure

**3. Collusion Attacks:**
- **Multiple watermarked models:** Each has unique watermark
- **Averaging attack:** Reduces all watermarks, detected by similarity drop
- **Mitigation:** Use orthogonal watermarks for different models


## Robustness Testing Design

### Attack Simulation Framework

**Purpose:** Systematically test watermark robustness against common attacks.

**Test Scenarios:**

1. **Pruning Attack:**
   ```python
   def simulate_pruning(model, prune_percentage):
       # Zero out random weights in classifier layer
       weights = model.classifier.weight.data
       mask = torch.rand_like(weights) > prune_percentage
       weights *= mask
       return model
   ```
   - Test pruning levels: 10%, 25%, 50%, 75%
   - Measure residual cosine similarity at each level
   - Expected: Detection fails when pruning > 50%

2. **Noise Injection Attack:**
   ```python
   def simulate_noise_injection(model, noise_std):
       # Add Gaussian noise to classifier weights
       weights = model.classifier.weight.data
       noise = torch.randn_like(weights) * noise_std
       weights += noise
       return model
   ```
   - Test noise levels: σ = 0.01, 0.05, 0.1, 0.2
   - Measure residual cosine similarity
   - Expected: Detection fails when σ > 0.1

3. **Fine-tuning Attack:**
   ```python
   def simulate_finetuning(model, dataloader, epochs):
       # Fine-tune model on CIFAR-10 subset
       optimizer = Adam(model.parameters(), lr=1e-4)
       for epoch in range(epochs):
           train_epoch(model, dataloader, optimizer)
       return model
   ```
   - Test fine-tuning durations: 1, 5, 10, 20 epochs
   - Measure weight deviation and residual similarity
   - Expected: Detection fails after 10+ epochs

4. **Model Compression:**
   - Test quantization (INT8, INT4)
   - Test knowledge distillation
   - Measure watermark survival rate

**Robustness Metrics:**
- **Survival Rate:** Percentage of attacks where cosine similarity > threshold
- **Degradation Curve:** Plot similarity vs attack intensity
- **Detection Boundary:** Minimum attack intensity causing detection failure

### Performance Benchmarking

**Metrics to Measure:**

1. **Training Performance:**
   - Time per epoch (with/without AWOA)
   - GPU memory usage
   - Convergence rate

2. **Watermarking Performance:**
   - Embedding time vs model size
   - Detection time vs model size
   - Memory overhead

3. **API Performance:**
   - Request latency (p50, p95, p99)
   - Throughput (requests/second)
   - Concurrent request handling

**Benchmark Suite:**
```python
def benchmark_embedding():
    model = load_model()
    watermark = generate_watermark()
    
    start = time.time()
    embed_watermark(model, watermark, alpha=0.1)
    end = time.time()
    
    assert (end - start) < 5.0  # Must complete in < 5 seconds
    
def benchmark_detection():
    model = load_watermarked_model()
    watermark = load_watermark()
    
    start = time.time()
    result = verify_watermark(model, watermark, alpha=0.1)
    end = time.time()
    
    assert (end - start) < 3.0  # Must complete in < 3 seconds
```

## Design Rationale & Tradeoffs

### 1. EfficientNet-B0 Selection

**Rationale:**
- Efficient architecture with good accuracy/parameter tradeoff
- Pretrained weights available for transfer learning
- Suitable for CIFAR-10 image size (32x32 → 224x224 resize)
- Widely used in production systems

**Tradeoffs:**
- Requires image resizing (32x32 → 224x224), increasing computation
- Alternative: Use smaller CNN designed for 32x32 images (faster but less accurate)
- Decision: Prioritize accuracy and real-world applicability

### 2. Classifier Layer Watermarking

**Rationale:**
- Classifier weights are task-specific and less likely to be reused
- Smaller embedding space (10×1280 = 12,800 values) enables fast operations
- Minimal impact on feature extraction layers

**Tradeoffs:**
- Watermark can be removed by replacing classifier layer
- Alternative: Embed in multiple layers (more robust but slower and more accuracy loss)
- Decision: Prioritize performance and simplicity for academic project

### 3. Additive Watermark Embedding

**Rationale:**
- Simple mathematical formula: W' = W + α×M
- Easy to extract: M = (W' - W) / α
- Well-studied in literature

**Tradeoffs:**
- Vulnerable to averaging attacks (multiple watermarked models)
- Alternative: Multiplicative embedding (more robust but harder to extract)
- Decision: Use additive for simplicity and reproducibility

### 4. Cosine Similarity Detection

**Rationale:**
- Scale-invariant (robust to weight scaling)
- Computationally efficient (O(n) complexity)
- Provides confidence metric in [-1, 1]

**Tradeoffs:**
- Sensitive to weight perturbations
- Alternative: Correlation coefficient, Euclidean distance
- Decision: Cosine similarity is standard in watermarking literature

### 5. AWOA for Hyperparameter Optimization

**Rationale:**
- Metaheuristic algorithm suitable for non-convex optimization
- Balances exploration and exploitation
- No gradient computation required

**Tradeoffs:**
- Computationally expensive (requires multiple training runs)
- Alternative: Grid search (exhaustive but slower), Bayesian optimization (faster but more complex)
- Decision: AWOA aligns with project requirements and provides research novelty

### 6. HMAC-SHA256 Authentication

**Rationale:**
- Industry-standard cryptographic authentication
- Collision-resistant and unforgeable
- Fast computation

**Tradeoffs:**
- Requires secure key management
- Alternative: Digital signatures (RSA, ECDSA) provide non-repudiation but slower
- Decision: HMAC sufficient for ownership verification without non-repudiation requirement

### 7. Flask API Architecture

**Rationale:**
- Lightweight and easy to develop
- Suitable for academic/research projects
- Good Python ecosystem integration

**Tradeoffs:**
- Not production-ready (single-threaded, no load balancing)
- Alternative: FastAPI (async support), Django (full-featured but heavier)
- Decision: Flask appropriate for project scope and timeline

### 8. File-Based Storage

**Rationale:**
- Simple implementation without database setup
- Direct access to model files for debugging
- Suitable for small-scale deployment

**Tradeoffs:**
- No ACID guarantees, limited scalability
- Alternative: Database (PostgreSQL, MongoDB) for metadata, S3 for files
- Decision: File-based storage sufficient for academic project with < 100 models

### 9. Alpha Parameter Range

**Rationale:**
- Range [0.01, 0.5] balances detectability and accuracy
- Lower bound (0.01): Minimal accuracy impact but weak detection
- Upper bound (0.5): Strong detection but significant accuracy loss

**Tradeoffs:**
- Narrow range limits robustness options
- Alternative: Adaptive alpha based on model sensitivity
- Decision: Fixed range simplifies validation and comparison

### 10. Binary Watermark Values

**Rationale:**
- Values in {-1, +1} maximize signal strength
- Simplifies generation and analysis
- Standard in digital watermarking

**Tradeoffs:**
- Less information capacity than real-valued watermarks
- Alternative: Gaussian watermarks (more capacity but harder to analyze)
- Decision: Binary watermarks sufficient for ownership verification


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

The following properties are derived from the requirements document and represent universal behaviors that must hold for all valid inputs. Each property is designed to be testable through property-based testing, where random inputs are generated to verify the property holds across a wide range of scenarios.

### Data Processing Properties

**Property 1: Normalization Correctness**

*For any* CIFAR-10 image tensor, applying the normalization transform should result in a tensor where each channel has been normalized using the specified mean and standard deviation values.

**Validates: Requirements 1.1**

**Property 2: Data Augmentation Application**

*For any* training image, when data augmentation is applied with a fixed random seed, the augmentation functions (horizontal flip, random crop) should be applied deterministically and produce consistent results across runs.

**Validates: Requirements 1.2, 1.3**

**Property 3: Batch Size Consistency**

*For any* DataLoader created with a specified batch size, all batches except possibly the last should have exactly that batch size, and the last batch should have size ≤ batch_size.

**Validates: Requirements 1.5**

**Property 4: Tensor Shape Preservation**

*For any* image in the preprocessed dataset, the output tensor should have shape (3, 224, 224) after resizing and transformation.

**Validates: Requirements 1.6**

### Model Training Properties

**Property 5: Model Architecture Initialization**

*For any* initialized EfficientNet-B0 model for CIFAR-10, the final classifier layer should have output dimension 10 and input dimension 1280.

**Validates: Requirements 2.1**

**Property 6: Weight Update During Training**

*For any* model and training batch, after performing one forward pass, loss computation, and backward pass with optimizer step, at least some model weights should differ from their initial values.

**Validates: Requirements 2.4**

**Property 7: Model Serialization Round-Trip**

*For any* trained model, saving the model to a .pth file and then loading it back should produce a model with identical weights (within floating-point precision).

**Validates: Requirements 2.6, 10.1**

### AWOA Optimization Properties

**Property 8: Population Initialization Bounds**

*For any* AWOA population initialization with a given search space, all whale agent positions should have hyperparameter values within the specified min/max bounds of the search space.

**Validates: Requirements 3.2**

**Property 9: Best Solution Selection**

*For any* completed AWOA optimization run, the returned best hyperparameters should have a fitness value greater than or equal to all other evaluated configurations.

**Validates: Requirements 3.5**

**Property 10: Iteration Count Compliance**

*For any* AWOA optimization with max_iterations parameter, the algorithm should perform exactly max_iterations iterations before returning results.

**Validates: Requirements 3.6**

### Watermark Embedding Properties

**Property 11: Watermark Shape Matching**

*For any* generated watermark for a model's classifier layer, the watermark shape should exactly match the shape of the classifier weight tensor.

**Validates: Requirements 4.1**

**Property 12: Embedding Formula Correctness**

*For any* model weights W, watermark M, and embedding strength α, the watermarked weights W' should satisfy: W'[i,j] = W[i,j] + α × M[i,j] for all positions (i,j).

**Validates: Requirements 4.2**

**Property 13: Non-Classifier Layer Preservation**

*For any* model before and after watermark embedding, all layers except the classifier layer should have identical weights (within floating-point precision).

**Validates: Requirements 4.5, 4.7**

**Property 14: Watermark Serialization Round-Trip**

*For any* watermark array, saving it to a .npy file and loading it back should produce an array with identical values and shape.

**Validates: Requirements 4.4, 10.2**

**Property 15: Deterministic Watermark Generation**

*For any* fixed random seed and shape, generating a watermark multiple times should produce identical arrays.

**Validates: Requirements 17.1**

### Watermark Detection Properties

**Property 16: Extraction Formula Correctness**

*For any* watermarked model with weights W', original weights W, and embedding strength α, the extracted watermark M_extracted should satisfy: M_extracted[i,j] = (W'[i,j] - W[i,j]) / α for all positions (i,j).

**Validates: Requirements 5.2**

**Property 17: Cosine Similarity Range**

*For any* two watermark arrays, the computed cosine similarity should be a value in the range [-1, 1].

**Validates: Requirements 5.3, 5.5**

**Property 18: Perfect Watermark Detection**

*For any* model with an embedded watermark that has not been modified, extracting the watermark and computing cosine similarity with the original should yield a value > 0.95.

**Validates: Requirements 5.4**

**Property 19: Threshold-Based Detection Decision**

*For any* watermark verification with a specified threshold, the detection status should be "pass" if and only if the cosine similarity is greater than or equal to the threshold.

**Validates: Requirements 5.4, 5.6**

### Alpha Validation Properties

**Property 20: Alpha Range Coverage**

*For any* alpha validation with a specified range of alpha values, the validation should test every alpha value in that range and return results for each.

**Validates: Requirements 6.1**

**Property 21: Accuracy Degradation Calculation**

*For any* base model accuracy A_base and watermarked model accuracy A_watermarked, the reported accuracy degradation should equal A_base - A_watermarked.

**Validates: Requirements 6.3**

**Property 22: Optimal Alpha Selection**

*For any* alpha validation results, the recommended optimal alpha should have accuracy degradation < 2% and cosine similarity > 0.95, and should maximize cosine similarity among valid candidates.

**Validates: Requirements 6.5**

### Authentication Properties

**Property 23: Certificate Structure Completeness**

*For any* generated ownership certificate, it should contain all required fields: owner_name, model_id, watermark_hash, timestamp, license_info, and hmac_signature.

**Validates: Requirements 7.1, 7.3**

**Property 24: HMAC Signature Verification Round-Trip**

*For any* valid ownership certificate generated with a secret key, verifying the certificate with the same secret key should return True, and verifying with a different key should return False.

**Validates: Requirements 7.2, 7.6**

**Property 25: Certificate Serialization Round-Trip**

*For any* ownership certificate, saving it to a .pkl file and loading it back should produce a certificate with identical field values.

**Validates: Requirements 7.4, 10.3**

**Property 26: Certificate Tampering Detection**

*For any* valid ownership certificate, modifying any field (except the signature) and then verifying should return False.

**Validates: Requirements 7.6, 14.3**

### License Management Properties

**Property 27: License Record Structure**

*For any* created license record, it should contain all required fields: license_id, model_id, licensee_name, license_type, issued_at, expires_at, restrictions, and status.

**Validates: Requirements 8.1**

**Property 28: License Status Computation**

*For any* license record with an expiration date, if the current date is before the expiration date and status is not "revoked", the queried status should be "active"; otherwise it should be "expired" or "revoked".

**Validates: Requirements 8.4**

**Property 29: License Update Persistence**

*For any* license record, updating the expiration date or restrictions and then querying the license should return the updated values.

**Validates: Requirements 8.5**

### Tamper Detection Properties

**Property 30: Pruning Detection**

*For any* model where classifier weights have been zeroed by more than 10%, the pruning detection should flag the model as pruned.

**Validates: Requirements 9.2**

**Property 31: Fine-Tuning Detection**

*For any* model where classifier weights have been modified such that the normalized L2 deviation from the original exceeds 10%, the fine-tuning detection should flag the model as fine-tuned.

**Validates: Requirements 9.4**

**Property 32: Residual Similarity Computation**

*For any* suspected tampered model, the tamper detection should compute and return a residual cosine similarity value between the extracted and original watermarks.

**Validates: Requirements 9.6**

**Property 33: Tamper Flagging**

*For any* model where the residual cosine similarity drops below the detection threshold, the system should flag the model as tampered.

**Validates: Requirements 9.7**

### Storage and Persistence Properties

**Property 34: Consistent File Naming**

*For any* saved watermarked model with identifier model_id, the model file, watermark file, and certificate file should all use model_id in their filenames following the naming convention.

**Validates: Requirements 10.4**

**Property 35: Registry Mapping Consistency**

*For any* registered model, the registry should contain a mapping from the model_id to the correct file paths for model, watermark, and certificate.

**Validates: Requirements 10.6**

**Property 36: Batch File Loading**

*For any* model_id in the registry, loading the model should return all three associated files (model, watermark, certificate) together.

**Validates: Requirements 10.7**

### API Properties

**Property 37: HTTP Status Code Correctness**

*For any* API request, if the request is valid and processing succeeds, the response should have status code 200; if the request has invalid parameters, status code 400; if a resource is not found, status code 404; if processing fails, status code 500.

**Validates: Requirements 11.6**

**Property 38: Error Response Format**

*For any* API request that results in an error, the response should be valid JSON containing at minimum a "status" field with value "error" and a "message" field with a descriptive error message.

**Validates: Requirements 11.7**

**Property 39: Input Validation Before Processing**

*For any* API request with invalid parameters (e.g., negative alpha, non-existent model_id), the system should reject the request with a 400 status code before attempting to process the operation.

**Validates: Requirements 11.8**

### Performance Properties

**Property 40: Accuracy Preservation**

*For any* model watermarked with the recommended alpha value, the accuracy degradation should be less than or equal to 2% of the base model accuracy.

**Validates: Requirements 13.1**

**Property 41: Embedding Time Bound**

*For any* EfficientNet-B0 model, watermark embedding should complete in less than 5 seconds on the specified hardware.

**Validates: Requirements 13.2**

**Property 42: Detection Time Bound**

*For any* watermark detection operation, the computation should complete in less than 3 seconds on the specified hardware.

**Validates: Requirements 13.3**

### Security Properties

**Property 43: Secret Key Confidentiality**

*For any* API response, the response body should not contain the secret key used for HMAC computation.

**Validates: Requirements 14.2**

**Property 44: Invalid Signature Rejection**

*For any* certificate with an invalid HMAC signature, the verification should return False and the certificate should be rejected.

**Validates: Requirements 14.4**

### Reliability Properties

**Property 45: Graceful Error Handling**

*For any* invalid input parameters provided to any system function, the function should return an error result or raise an exception with a descriptive message, rather than crashing or producing undefined behavior.

**Validates: Requirements 15.3**

**Property 46: Operation Logging**

*For any* watermarking operation (embed, detect, certificate generation), the system should create a log entry with a timestamp.

**Validates: Requirements 15.4**

### Scalability Properties

**Property 47: Concurrent Request Safety**

*For any* set of concurrent API requests that do not conflict (e.g., operating on different models), processing them concurrently should produce the same results as processing them sequentially.

**Validates: Requirements 15.5, 16.1**

### Reproducibility Properties

**Property 48: AWOA Determinism**

*For any* AWOA optimization run with a fixed random seed, running the optimization multiple times should produce identical hyperparameter recommendations.

**Validates: Requirements 17.3**

**Property 49: Hyperparameter Documentation**

*For any* trained model or watermarked model, the system should record and store all hyperparameters used in training and embedding.

**Validates: Requirements 17.2**

### Usability Properties

**Property 50: Default Parameter Provision**

*For any* API endpoint or function with optional parameters, the system should provide sensible default values that allow the operation to succeed without requiring the user to specify every parameter.

**Validates: Requirements 18.3**

**Property 51: Error Message Actionability**

*For any* error that occurs due to invalid user input, the error message should include a suggestion for how to correct the input.

**Validates: Requirements 18.4**


## Error Handling

### Error Categories

The system implements comprehensive error handling across four categories:

1. **Input Validation Errors:** Invalid parameters, malformed data
2. **Resource Errors:** Missing files, insufficient memory/disk space
3. **Computation Errors:** Numerical instability, convergence failures
4. **Security Errors:** Invalid signatures, unauthorized access

### Error Handling Strategy

**Principle:** Fail fast with descriptive error messages and suggested corrective actions.

### Input Validation Errors

**Scenario:** Invalid alpha value (outside range [0.01, 0.5])

```python
class InvalidAlphaError(ValueError):
    def __init__(self, alpha, min_alpha=0.01, max_alpha=0.5):
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        message = (
            f"Invalid embedding strength alpha={alpha}. "
            f"Must be in range [{min_alpha}, {max_alpha}]. "
            f"Suggestion: Use alpha=0.1 for balanced robustness and accuracy."
        )
        super().__init__(message)
```

**Scenario:** Model architecture mismatch

```python
class ArchitectureMismatchError(ValueError):
    def __init__(self, expected, actual):
        message = (
            f"Model architecture mismatch. Expected {expected}, got {actual}. "
            f"Suggestion: Ensure you are using EfficientNet-B0 trained on CIFAR-10."
        )
        super().__init__(message)
```

**Scenario:** Watermark shape mismatch

```python
class WatermarkShapeMismatchError(ValueError):
    def __init__(self, watermark_shape, weight_shape):
        message = (
            f"Watermark shape {watermark_shape} does not match "
            f"classifier weight shape {weight_shape}. "
            f"Suggestion: Generate watermark with correct shape using "
            f"generate_watermark(shape=model.classifier.weight.shape)."
        )
        super().__init__(message)
```

### Resource Errors

**Scenario:** Model file not found

```python
class ModelNotFoundError(FileNotFoundError):
    def __init__(self, model_id):
        message = (
            f"Model '{model_id}' not found in registry. "
            f"Suggestion: Check available models using list_models() or "
            f"train a new model using train_model()."
        )
        super().__init__(message)
```

**Scenario:** Insufficient GPU memory

```python
class InsufficientMemoryError(RuntimeError):
    def __init__(self, required_mb, available_mb):
        message = (
            f"Insufficient GPU memory. Required: {required_mb}MB, "
            f"Available: {available_mb}MB. "
            f"Suggestion: Reduce batch size or use CPU training."
        )
        super().__init__(message)
```

**Scenario:** Disk space exhausted

```python
class DiskSpaceError(IOError):
    def __init__(self, required_mb, available_mb):
        message = (
            f"Insufficient disk space. Required: {required_mb}MB, "
            f"Available: {available_mb}MB. "
            f"Suggestion: Free up disk space or change storage_root location."
        )
        super().__init__(message)
```

### Computation Errors

**Scenario:** AWOA convergence failure

```python
class ConvergenceError(RuntimeError):
    def __init__(self, iterations, best_fitness):
        message = (
            f"AWOA failed to converge after {iterations} iterations. "
            f"Best fitness achieved: {best_fitness}. "
            f"Suggestion: Increase max_iterations or adjust search_space bounds."
        )
        super().__init__(message)
```

**Scenario:** Numerical instability in cosine similarity

```python
class NumericalInstabilityError(RuntimeError):
    def __init__(self, norm_value):
        message = (
            f"Numerical instability detected. Norm value: {norm_value}. "
            f"Suggestion: Check for NaN or Inf values in watermark or weights."
        )
        super().__init__(message)
```

### Security Errors

**Scenario:** Invalid HMAC signature

```python
class InvalidSignatureError(SecurityError):
    def __init__(self, certificate_id):
        message = (
            f"Invalid HMAC signature for certificate '{certificate_id}'. "
            f"Certificate may have been tampered with. "
            f"Suggestion: Verify certificate source and regenerate if necessary."
        )
        super().__init__(message)
        # Log security event
        logger.warning(f"Invalid signature attempt for certificate {certificate_id}")
```

**Scenario:** Secret key not found

```python
class SecretKeyError(SecurityError):
    def __init__(self, key_path):
        message = (
            f"Secret key not found at '{key_path}'. "
            f"Suggestion: Generate a new secret key using generate_secret_key() "
            f"and store it securely."
        )
        super().__init__(message)
```

### API Error Responses

**Standard Error Response Format:**

```python
def create_error_response(error_code, message, details=None, status_code=400):
    response = {
        "status": "error",
        "error_code": error_code,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    if details:
        response["details"] = details
    return jsonify(response), status_code
```

**Example Error Responses:**

```json
{
  "status": "error",
  "error_code": "INVALID_ALPHA",
  "message": "Invalid embedding strength alpha=0.8",
  "details": "Must be in range [0.01, 0.5]. Suggestion: Use alpha=0.1",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

```json
{
  "status": "error",
  "error_code": "MODEL_NOT_FOUND",
  "message": "Model 'efficientnet_b0_999' not found",
  "details": "Check available models using GET /models",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Rollback Mechanisms

**Transaction Pattern for File Operations:**

```python
def save_watermarked_model_transactional(model, watermark, certificate, model_id):
    # Create temporary files
    temp_model_path = f"{model_id}.pth.tmp"
    temp_watermark_path = f"{model_id}.npy.tmp"
    temp_certificate_path = f"{model_id}.pkl.tmp"
    
    try:
        # Save to temporary files
        torch.save(model.state_dict(), temp_model_path)
        np.save(temp_watermark_path, watermark)
        with open(temp_certificate_path, 'wb') as f:
            pickle.dump(certificate, f)
        
        # Verify integrity
        verify_file_integrity(temp_model_path)
        verify_file_integrity(temp_watermark_path)
        verify_file_integrity(temp_certificate_path)
        
        # Atomic rename (move to final location)
        os.rename(temp_model_path, f"{model_id}.pth")
        os.rename(temp_watermark_path, f"{model_id}.npy")
        os.rename(temp_certificate_path, f"{model_id}.pkl")
        
        # Update registry
        update_registry(model_id, ...)
        
    except Exception as e:
        # Rollback: delete temporary files
        for temp_file in [temp_model_path, temp_watermark_path, temp_certificate_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise
```

### Logging Strategy

**Log Levels:**
- **DEBUG:** Detailed diagnostic information (hyperparameters, intermediate values)
- **INFO:** General operational events (model trained, watermark embedded)
- **WARNING:** Unexpected but recoverable events (low cosine similarity, convergence issues)
- **ERROR:** Error events that prevent operation completion
- **CRITICAL:** Security events (invalid signatures, unauthorized access)

**Log Format:**

```
[2024-01-15 10:30:00] [INFO] [ModelTraining] Model efficientnet_b0_001 trained successfully. Accuracy: 0.89
[2024-01-15 10:31:00] [INFO] [WatermarkEmbedding] Watermark embedded in model efficientnet_b0_001. Alpha: 0.1, Similarity: 0.98
[2024-01-15 10:32:00] [WARNING] [WatermarkDetection] Low cosine similarity detected: 0.87 (threshold: 0.95)
[2024-01-15 10:33:00] [ERROR] [API] Invalid alpha value: 0.8. Expected range: [0.01, 0.5]
[2024-01-15 10:34:00] [CRITICAL] [Authentication] Invalid HMAC signature for certificate cert_001
```

## Testing Strategy

### Dual Testing Approach

The system employs both unit testing and property-based testing to ensure comprehensive correctness validation:

- **Unit Tests:** Verify specific examples, edge cases, and error conditions
- **Property Tests:** Verify universal properties across all inputs through randomized testing

Both approaches are complementary and necessary. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across a wide input space.

### Property-Based Testing

**Framework:** Use `hypothesis` library for Python property-based testing

**Configuration:**
- Minimum 100 iterations per property test (due to randomization)
- Each test tagged with feature name and property number
- Tag format: `# Feature: ai-model-watermarking, Property N: [property description]`

**Example Property Test:**

```python
from hypothesis import given, strategies as st
import hypothesis.strategies as st

@given(
    alpha=st.floats(min_value=0.01, max_value=0.5),
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100)
def test_property_12_embedding_formula_correctness(alpha, seed):
    """
    Feature: ai-model-watermarking, Property 12: Embedding Formula Correctness
    
    For any model weights W, watermark M, and embedding strength α,
    the watermarked weights W' should satisfy: W'[i,j] = W[i,j] + α × M[i,j]
    """
    # Setup
    model = create_test_model()
    W_original = model.classifier.weight.data.clone()
    watermark = generate_watermark(W_original.shape, seed=seed)
    
    # Execute
    embed_watermark(model, watermark, alpha)
    W_watermarked = model.classifier.weight.data
    
    # Verify
    expected = W_original + alpha * torch.tensor(watermark)
    assert torch.allclose(W_watermarked, expected, rtol=1e-5)
```

**Example Property Test for Round-Trip:**

```python
@given(
    alpha=st.floats(min_value=0.01, max_value=0.5),
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100)
def test_property_18_perfect_watermark_detection(alpha, seed):
    """
    Feature: ai-model-watermarking, Property 18: Perfect Watermark Detection
    
    For any model with an embedded watermark that has not been modified,
    extracting the watermark and computing cosine similarity with the original
    should yield a value > 0.95.
    """
    # Setup
    model = create_test_model()
    watermark = generate_watermark(model.classifier.weight.shape, seed=seed)
    
    # Embed watermark
    embed_watermark(model, watermark, alpha)
    
    # Detect watermark
    result = verify_watermark(model, watermark, alpha, threshold=0.95)
    
    # Verify
    assert result['cosine_similarity'] > 0.95
    assert result['detection_status'] == 'pass'
```

### Unit Testing

**Framework:** Use `pytest` for unit testing

**Coverage Target:** Minimum 80% code coverage for core watermarking functions

**Test Categories:**

1. **Smoke Tests:** Basic functionality works
2. **Edge Case Tests:** Boundary conditions, empty inputs, extreme values
3. **Error Tests:** Invalid inputs trigger appropriate errors
4. **Integration Tests:** Components work together correctly

**Example Unit Tests:**

```python
def test_dataset_loading():
    """Test CIFAR-10 dataset loads with correct split sizes."""
    train_dataset = load_cifar10(train=True)
    test_dataset = load_cifar10(train=False)
    
    assert len(train_dataset) == 50000
    assert len(test_dataset) == 10000

def test_model_initialization():
    """Test EfficientNet-B0 initializes with correct output dimension."""
    model = initialize_model(num_classes=10)
    
    assert model.classifier.out_features == 10
    assert model.classifier.in_features == 1280

def test_watermark_embedding_edge_case_zero_alpha():
    """Test watermark embedding with alpha=0 leaves weights unchanged."""
    model = create_test_model()
    W_original = model.classifier.weight.data.clone()
    watermark = generate_watermark(W_original.shape)
    
    embed_watermark(model, watermark, alpha=0.0)
    W_after = model.classifier.weight.data
    
    assert torch.allclose(W_original, W_after)

def test_invalid_alpha_raises_error():
    """Test that invalid alpha value raises appropriate error."""
    model = create_test_model()
    watermark = generate_watermark(model.classifier.weight.shape)
    
    with pytest.raises(InvalidAlphaError) as exc_info:
        embed_watermark(model, watermark, alpha=0.8)
    
    assert "Must be in range [0.01, 0.5]" in str(exc_info.value)

def test_hmac_signature_verification_round_trip():
    """Test HMAC signature generation and verification."""
    secret_key = generate_secret_key()
    certificate = generate_certificate(
        owner_name="Test Owner",
        model_id="test_model_001",
        watermark_hash="abc123",
        license_info={}
    )
    
    # Valid signature should verify
    assert verify_certificate(certificate, secret_key) == True
    
    # Modified certificate should fail verification
    certificate['owner_name'] = "Modified Owner"
    assert verify_certificate(certificate, secret_key) == False
```

### Integration Testing

**Purpose:** Verify end-to-end workflows function correctly

**Test Scenarios:**

1. **Complete Watermarking Workflow:**
   - Train model → Embed watermark → Save files → Load files → Verify watermark
   
2. **Alpha Validation Workflow:**
   - Train model → Test multiple alphas → Select optimal → Embed with optimal → Verify

3. **Tamper Detection Workflow:**
   - Embed watermark → Apply attack → Detect tampering → Verify detection

4. **API Workflow:**
   - POST /train → POST /embed → POST /verify → GET /models

**Example Integration Test:**

```python
def test_complete_watermarking_workflow():
    """Test complete workflow from training to verification."""
    # Train model
    model = train_model(hyperparams={'learning_rate': 0.001, 'epochs': 5})
    model_id = "integration_test_001"
    
    # Generate and embed watermark
    watermark = generate_watermark(model.classifier.weight.shape, seed=42)
    embed_watermark(model, watermark, alpha=0.1)
    
    # Generate certificate
    certificate = generate_certificate(
        owner_name="Integration Test",
        model_id=model_id,
        watermark_hash=compute_hash(watermark),
        license_info={"type": "test"}
    )
    
    # Save all files
    save_watermarked_model(model, watermark, certificate, model_id)
    
    # Load files
    loaded_model, loaded_watermark, loaded_certificate = load_watermarked_model(model_id)
    
    # Verify watermark
    result = verify_watermark(loaded_model, loaded_watermark, alpha=0.1, threshold=0.95)
    
    # Verify certificate
    cert_valid = verify_certificate(loaded_certificate)
    
    # Assertions
    assert result['detection_status'] == 'pass'
    assert result['cosine_similarity'] > 0.95
    assert cert_valid == True
```

### Performance Testing

**Benchmarks:**

```python
def test_embedding_performance():
    """Test watermark embedding completes within time bound."""
    model = create_test_model()
    watermark = generate_watermark(model.classifier.weight.shape)
    
    start = time.time()
    embed_watermark(model, watermark, alpha=0.1)
    duration = time.time() - start
    
    assert duration < 5.0  # Must complete in < 5 seconds

def test_detection_performance():
    """Test watermark detection completes within time bound."""
    model = create_watermarked_model()
    watermark = load_test_watermark()
    
    start = time.time()
    verify_watermark(model, watermark, alpha=0.1, threshold=0.95)
    duration = time.time() - start
    
    assert duration < 3.0  # Must complete in < 3 seconds
```

### Security Testing

**Test Scenarios:**

1. **Certificate Forgery Attempt:**
   - Modify certificate fields
   - Verify signature validation fails

2. **Watermark Removal Attempt:**
   - Apply pruning/noise/fine-tuning
   - Verify tamper detection succeeds

3. **Key Exposure Prevention:**
   - Make API requests
   - Verify secret key not in responses

**Example Security Test:**

```python
def test_certificate_tampering_detection():
    """Test that modified certificates are detected as invalid."""
    secret_key = generate_secret_key()
    certificate = generate_certificate(
        owner_name="Original Owner",
        model_id="test_001",
        watermark_hash="abc123",
        license_info={}
    )
    
    # Original certificate should verify
    assert verify_certificate(certificate, secret_key) == True
    
    # Tamper with certificate
    certificate['owner_name'] = "Attacker"
    
    # Tampered certificate should fail verification
    assert verify_certificate(certificate, secret_key) == False
```

### Test Execution

**Continuous Integration:**
- Run all unit tests on every commit
- Run property tests nightly (due to longer execution time)
- Run integration tests before releases
- Generate coverage reports

**Test Commands:**

```bash
# Run all unit tests
pytest tests/unit/ -v --cov=src --cov-report=html

# Run property tests with 100 examples
pytest tests/property/ -v --hypothesis-show-statistics

# Run integration tests
pytest tests/integration/ -v

# Run performance benchmarks
pytest tests/performance/ -v --benchmark-only

# Run security tests
pytest tests/security/ -v
```

### Test Data Management

**Fixtures:**
- Pre-trained test models (small, fast to load)
- Sample watermarks with known properties
- Test certificates with valid/invalid signatures
- CIFAR-10 subset for fast testing (1000 images)

**Data Generation:**
- Use `hypothesis` strategies for random test data
- Use fixed seeds for reproducible tests
- Generate edge cases programmatically

---

## Summary

This design document provides a comprehensive blueprint for implementing the AI Model Watermarking System. The architecture balances simplicity, security, and performance while meeting all functional and non-functional requirements. The correctness properties ensure that the system behaves correctly across all valid inputs, and the dual testing strategy (unit + property-based) provides strong correctness guarantees.

Key design decisions prioritize academic project constraints while maintaining real-world applicability. The modular architecture enables independent development and testing of components, and the comprehensive error handling ensures robustness in production use.
