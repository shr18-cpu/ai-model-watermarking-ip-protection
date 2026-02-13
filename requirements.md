# Requirements Document: AI Model Watermarking System

## Introduction

This document specifies the requirements for an AI Model Watermarking System designed to protect intellectual property rights of trained deep learning models. The system embeds digital watermarks into model weights, enables ownership verification, tracks licensing information, and detects tampering attempts while preserving model performance.

The system addresses the critical problem of unauthorized copying, redistribution, and misuse of deep learning models by providing cryptographic proof of ownership and detecting malicious modifications such as pruning, noise injection, and fine-tuning attacks.

## Project Overview

**Title:** Protecting Intellectual Property in Artificial Intelligence Models using Digital Watermarking

**Purpose:** Develop a secure AI watermarking framework that protects ownership, authenticity, originality, and usage rights of trained deep learning models through embedded digital watermarks and cryptographic authentication.

**Technical Foundation:**
- Base Architecture: EfficientNet-B0 deep convolutional neural network
- Training Dataset: CIFAR-10 (60,000 32x32 color images in 10 classes)
- Optimization: Adaptive Whale Optimization Algorithm (AWOA) for hyperparameter tuning
- Watermarking Method: Binary digital watermark embedded in classifier layer weights
- Detection Method: Cosine similarity-based watermark extraction and verification
- Authentication: HMAC-SHA256 for ownership certificate generation
- Storage Formats: PyTorch model (.pth), NumPy watermark (.npy), Pickle certificate (.pkl)
- Deployment: Flask REST API backend with HTML/CSS/JavaScript frontend

## Scope

### In Scope

- Digital watermark embedding in EfficientNet-B0 classifier weights
- Ownership verification through correlation-based watermark detection
- License information management and tracking
- Tamper detection for pruning, noise injection, and fine-tuning attacks
- AWOA-based hyperparameter optimization for model training
- CIFAR-10 dataset preprocessing and augmentation
- Flask API for watermarking operations
- Web-based user interface for system interaction
- Model persistence and retrieval mechanisms
- Performance metrics tracking and reporting

### Out of Scope

- Watermarking of models other than EfficientNet-B0
- Datasets other than CIFAR-10
- Real-time inference watermarking
- Distributed watermarking across multiple models
- Blockchain-based ownership tracking
- Legal enforcement mechanisms
- Commercial licensing platform integration

## Stakeholders

- **Model Developers:** Create and train deep learning models requiring IP protection
- **Model Owners:** Hold intellectual property rights and require ownership verification
- **Licensees:** Authorized users who deploy watermarked models under license agreements
- **System Administrators:** Maintain and operate the watermarking infrastructure
- **Security Auditors:** Verify tamper detection capabilities and authentication mechanisms
- **Research Community:** Academic users evaluating watermarking effectiveness

## Glossary

- **System:** The AI Model Watermarking System
- **Watermark:** A binary digital signature embedded in model weights
- **Base_Model:** The unwatermarked EfficientNet-B0 model trained on CIFAR-10
- **Watermarked_Model:** The EfficientNet-B0 model with embedded watermark
- **AWOA:** Adaptive Whale Optimization Algorithm for hyperparameter tuning
- **Embedding_Strength:** The alpha parameter controlling watermark intensity
- **Ownership_Certificate:** HMAC-SHA256 authenticated document linking watermark to owner
- **Cosine_Similarity:** Metric for measuring watermark detection confidence
- **Tamper_Attack:** Malicious modification including pruning, noise injection, or fine-tuning
- **Classifier_Weights:** The final fully-connected layer weights where watermark is embedded
- **License_Record:** Metadata associating watermark with usage permissions
- **Detection_Threshold:** Minimum cosine similarity value for successful watermark detection

## Functional Requirements

### Requirement 1: Dataset Preprocessing and Preparation

**User Story:** As a model developer, I want to preprocess the CIFAR-10 dataset with appropriate transformations, so that the base model can be trained with high-quality normalized data.

#### Acceptance Criteria

1. WHEN CIFAR-10 dataset is loaded, THE System SHALL apply normalization with mean [0.4914, 0.4822, 0.4465] and standard deviation [0.2470, 0.2435, 0.2616]
2. WHEN training data is prepared, THE System SHALL apply random horizontal flip augmentation with probability 0.5
3. WHEN training data is prepared, THE System SHALL apply random crop with padding of 4 pixels
4. THE System SHALL split CIFAR-10 into 50,000 training images and 10,000 test images
5. WHEN data loaders are created, THE System SHALL use batch size determined by AWOA optimization
6. THE System SHALL convert all images to PyTorch tensors with shape (3, 32, 32)

### Requirement 2: Baseline Model Training

**User Story:** As a model developer, I want to train an EfficientNet-B0 model on CIFAR-10 with optimized hyperparameters, so that I have a high-performing base model for watermark embedding.

#### Acceptance Criteria

1. THE System SHALL initialize EfficientNet-B0 architecture with 10 output classes for CIFAR-10
2. WHEN training begins, THE System SHALL use hyperparameters optimized by AWOA
3. WHEN training an epoch, THE System SHALL compute cross-entropy loss for all batches
4. WHEN backpropagation occurs, THE System SHALL update model weights using the optimizer
5. WHEN an epoch completes, THE System SHALL evaluate model accuracy on the test set
6. THE System SHALL save the trained model weights to a .pth file
7. WHEN training completes, THE System SHALL report final test accuracy and loss metrics

### Requirement 3: AWOA Hyperparameter Optimization

**User Story:** As a model developer, I want to optimize training hyperparameters using Adaptive Whale Optimization Algorithm, so that the base model achieves maximum performance before watermarking.

#### Acceptance Criteria

1. THE System SHALL optimize learning rate, batch size, and number of epochs using AWOA
2. WHEN AWOA initializes, THE System SHALL create a population of whale agents with random hyperparameter values
3. WHEN evaluating a whale agent, THE System SHALL train the model with those hyperparameters and return validation accuracy as fitness
4. WHEN updating whale positions, THE System SHALL apply spiral updating mechanism and random search based on adaptive parameters
5. WHEN AWOA completes, THE System SHALL return the hyperparameter set with highest validation accuracy
6. THE System SHALL execute AWOA for a configurable number of iterations

### Requirement 4: Watermark Generation and Embedding

**User Story:** As a model owner, I want to embed a unique binary watermark into my trained model's classifier weights, so that I can prove ownership and detect unauthorized copies.

#### Acceptance Criteria

1. THE System SHALL generate a binary watermark as a random array matching the shape of classifier weights
2. WHEN embedding a watermark, THE System SHALL modify classifier weights using the formula: watermarked_weights = original_weights + alpha * watermark
3. THE System SHALL accept embedding strength (alpha) as a configurable parameter
4. WHEN watermark is embedded, THE System SHALL save the original watermark array to a .npy file
5. THE System SHALL embed watermark only in the final fully-connected classifier layer
6. WHEN embedding completes, THE System SHALL save the Watermarked_Model to a .pth file
7. THE System SHALL preserve all non-classifier layers unchanged during embedding

### Requirement 5: Watermark Detection and Extraction

**User Story:** As a model owner, I want to extract and verify the watermark from a suspected model, so that I can confirm ownership and detect unauthorized usage.

#### Acceptance Criteria

1. WHEN detecting a watermark, THE System SHALL load the Watermarked_Model and original Watermark
2. THE System SHALL extract the embedded watermark using the formula: extracted_watermark = (watermarked_weights - original_weights) / alpha
3. WHEN comparing watermarks, THE System SHALL compute Cosine_Similarity between original and extracted watermarks
4. THE System SHALL report watermark detection as successful when Cosine_Similarity exceeds Detection_Threshold
5. THE System SHALL return Cosine_Similarity value as a confidence metric
6. WHEN watermark detection fails, THE System SHALL return Cosine_Similarity below Detection_Threshold

### Requirement 6: Embedding Strength Validation

**User Story:** As a model owner, I want to validate different embedding strength values, so that I can select an alpha that balances watermark robustness and model performance.

#### Acceptance Criteria

1. THE System SHALL test multiple alpha values in a configurable range
2. WHEN testing an alpha value, THE System SHALL embed watermark, evaluate model accuracy, and compute detection confidence
3. THE System SHALL report accuracy degradation as the difference between Base_Model and Watermarked_Model accuracy
4. THE System SHALL report detection confidence as Cosine_Similarity for each alpha value
5. WHEN validation completes, THE System SHALL recommend the alpha value with acceptable accuracy loss and high detection confidence
6. THE System SHALL generate a validation report showing alpha vs accuracy and alpha vs detection confidence curves

### Requirement 7: Ownership Certificate Generation

**User Story:** As a model owner, I want to generate a cryptographically authenticated ownership certificate, so that I can prove legal ownership of the watermarked model.

#### Acceptance Criteria

1. WHEN generating an Ownership_Certificate, THE System SHALL create a record containing owner name, model identifier, watermark hash, timestamp, and license terms
2. THE System SHALL compute HMAC-SHA256 signature using a secret key and certificate data
3. THE System SHALL include the HMAC signature in the Ownership_Certificate
4. WHEN saving the certificate, THE System SHALL serialize it to a .pkl file
5. THE System SHALL link the Ownership_Certificate to the corresponding Watermark file
6. WHEN verifying a certificate, THE System SHALL recompute HMAC and compare with stored signature

### Requirement 8: License Management and Tracking

**User Story:** As a model owner, I want to associate license information with watermarked models, so that I can track authorized usage and detect license violations.

#### Acceptance Criteria

1. THE System SHALL store License_Record containing licensee name, license type, expiration date, and usage restrictions
2. WHEN creating a License_Record, THE System SHALL link it to the corresponding Watermark and Ownership_Certificate
3. THE System SHALL support license types including commercial, academic, and evaluation
4. WHEN querying a license, THE System SHALL return license status as active, expired, or revoked
5. THE System SHALL allow updating license expiration dates and usage restrictions
6. THE System SHALL maintain a log of all license queries and modifications

### Requirement 9: Tamper Detection

**User Story:** As a model owner, I want to detect if my watermarked model has been tampered with through pruning, noise injection, or fine-tuning, so that I can identify unauthorized modifications.

#### Acceptance Criteria

1. WHEN detecting tampering, THE System SHALL compare Watermarked_Model architecture with expected EfficientNet-B0 structure
2. THE System SHALL detect pruning by identifying removed or zeroed weights in classifier layer
3. WHEN noise injection is suspected, THE System SHALL compute weight distribution statistics and compare with original model
4. THE System SHALL detect fine-tuning by measuring weight deviation from original Watermarked_Model
5. WHEN tampering is detected, THE System SHALL report the type of attack and confidence level
6. THE System SHALL compute residual Cosine_Similarity after suspected tampering
7. WHEN Cosine_Similarity drops below Detection_Threshold after modification, THE System SHALL flag the model as tampered

### Requirement 10: Model Persistence and Retrieval

**User Story:** As a system administrator, I want to save and load watermarked models with associated metadata, so that the system can manage multiple watermarked models efficiently.

#### Acceptance Criteria

1. WHEN saving a Watermarked_Model, THE System SHALL store model weights in .pth format
2. THE System SHALL save the original Watermark in .npy format
3. THE System SHALL save the Ownership_Certificate in .pkl format
4. WHEN saving files, THE System SHALL use a consistent naming convention linking model, watermark, and certificate
5. WHEN loading a Watermarked_Model, THE System SHALL verify file integrity before loading
6. THE System SHALL maintain a registry mapping model identifiers to file paths
7. WHEN retrieving a model, THE System SHALL load all associated files (model, watermark, certificate) together

### Requirement 11: Backend API Services

**User Story:** As a frontend developer, I want REST API endpoints for all watermarking operations, so that I can build a user interface for the system.

#### Acceptance Criteria

1. THE System SHALL provide a Flask REST API with endpoints for all watermarking operations
2. WHEN receiving a training request, THE System SHALL accept dataset parameters and return trained Base_Model identifier
3. WHEN receiving an embedding request, THE System SHALL accept model identifier, watermark, and alpha value, then return Watermarked_Model identifier
4. WHEN receiving a detection request, THE System SHALL accept model identifier and watermark, then return Cosine_Similarity and detection status
5. WHEN receiving a certificate request, THE System SHALL accept owner information and return Ownership_Certificate
6. THE System SHALL return appropriate HTTP status codes for success, client errors, and server errors
7. WHEN an API error occurs, THE System SHALL return JSON error messages with descriptive details
8. THE System SHALL validate all API request parameters before processing
9. THE System SHALL implement CORS headers for cross-origin requests from the frontend

### Requirement 12: Frontend User Interface

**User Story:** As a model owner, I want a web-based interface to train models, embed watermarks, and verify ownership, so that I can use the system without command-line tools.

#### Acceptance Criteria

1. THE System SHALL provide an HTML interface with forms for training, embedding, detection, and certificate generation
2. WHEN a user submits a training request, THE System SHALL display progress indicators and show results upon completion
3. WHEN a user uploads a model for watermark embedding, THE System SHALL validate file format and size before processing
4. WHEN displaying detection results, THE System SHALL show Cosine_Similarity value and visual indicators for pass/fail status
5. THE System SHALL display ownership certificates in human-readable format
6. WHEN an operation fails, THE System SHALL display error messages to the user
7. THE System SHALL provide download links for generated models, watermarks, and certificates

## Non-Functional Requirements

### Requirement 13: Performance Preservation

**User Story:** As a model owner, I want watermark embedding to minimally impact model accuracy, so that my watermarked model remains useful for its intended task.

#### Acceptance Criteria

1. WHEN a watermark is embedded with recommended alpha value, THE System SHALL maintain model accuracy within 2% of Base_Model accuracy
2. THE System SHALL complete watermark embedding in under 5 seconds for EfficientNet-B0
3. THE System SHALL complete watermark detection in under 3 seconds
4. WHEN inference is performed on Watermarked_Model, THE System SHALL maintain inference latency within 5% of Base_Model latency

### Requirement 14: Security and Authentication

**User Story:** As a model owner, I want strong cryptographic protection for ownership certificates, so that attackers cannot forge ownership claims.

#### Acceptance Criteria

1. THE System SHALL use HMAC-SHA256 with minimum 256-bit secret keys for certificate authentication
2. WHEN storing secret keys, THE System SHALL use secure storage mechanisms and never expose keys in API responses
3. THE System SHALL validate HMAC signatures before accepting any Ownership_Certificate as authentic
4. WHEN an invalid signature is detected, THE System SHALL reject the certificate and log the attempt
5. THE System SHALL use secure random number generation for watermark creation

### Requirement 15: Reliability and Robustness

**User Story:** As a system administrator, I want the system to handle errors gracefully and maintain data integrity, so that operations do not corrupt models or lose watermarks.

#### Acceptance Criteria

1. WHEN a file operation fails, THE System SHALL rollback partial changes and maintain previous state
2. THE System SHALL validate model architecture before attempting watermark embedding
3. WHEN invalid parameters are provided, THE System SHALL return descriptive error messages without crashing
4. THE System SHALL log all operations with timestamps for audit trails
5. WHEN concurrent requests occur, THE System SHALL handle them safely without data corruption

### Requirement 16: Scalability

**User Story:** As a system administrator, I want the system to handle multiple watermarking operations efficiently, so that it can serve multiple users concurrently.

#### Acceptance Criteria

1. THE System SHALL support concurrent API requests from multiple clients
2. WHEN processing multiple embedding requests, THE System SHALL queue operations and process them sequentially
3. THE System SHALL maintain response times under 10 seconds for 95% of requests under normal load
4. THE System SHALL handle at least 100 stored watermarked models without performance degradation

### Requirement 17: Reproducibility

**User Story:** As a researcher, I want deterministic watermark generation and embedding, so that I can reproduce experimental results.

#### Acceptance Criteria

1. WHEN a random seed is provided, THE System SHALL generate identical watermarks across multiple runs
2. THE System SHALL document all hyperparameters used in model training and watermark embedding
3. WHEN AWOA optimization is run with a fixed seed, THE System SHALL produce identical hyperparameter recommendations
4. THE System SHALL log software versions and dependencies for reproducibility

### Requirement 18: Usability

**User Story:** As a model owner with limited technical expertise, I want clear documentation and intuitive interfaces, so that I can use the system without extensive training.

#### Acceptance Criteria

1. THE System SHALL provide user documentation explaining all operations with examples
2. WHEN displaying results, THE System SHALL use clear labels and units for all metrics
3. THE System SHALL provide default parameter values for all optional inputs
4. WHEN errors occur, THE System SHALL suggest corrective actions in error messages

## System Requirements

### Software Requirements

- **Programming Language:** Python 3.8 or higher
- **Deep Learning Framework:** PyTorch 1.10 or higher with torchvision
- **Web Framework:** Flask 2.0 or higher
- **Numerical Computing:** NumPy 1.21 or higher
- **Optimization Library:** Custom AWOA implementation or equivalent metaheuristic library
- **Cryptography:** Python hashlib and hmac standard libraries
- **Frontend:** Modern web browser supporting HTML5, CSS3, and ES6 JavaScript
- **Operating System:** Linux, macOS, or Windows with Python support

### Hardware Requirements

- **Minimum:** 8 GB RAM, 4-core CPU, 10 GB storage
- **Recommended:** 16 GB RAM, 8-core CPU, GPU with 4 GB VRAM (CUDA-compatible), 50 GB storage
- **Training:** GPU acceleration strongly recommended for AWOA optimization and model training

## Data Requirements

### Input Data

- **Dataset:** CIFAR-10 (automatically downloaded via torchvision)
- **Model Files:** PyTorch .pth files containing EfficientNet-B0 state dictionaries
- **Watermark Files:** NumPy .npy files containing binary watermark arrays
- **Certificate Files:** Pickle .pkl files containing ownership certificate dictionaries

### Output Data

- **Watermarked Models:** PyTorch .pth files with embedded watermarks
- **Watermarks:** NumPy .npy files with binary watermark arrays
- **Ownership Certificates:** Pickle .pkl files with HMAC-authenticated ownership records
- **Validation Reports:** JSON or CSV files with alpha validation results
- **Detection Reports:** JSON files with cosine similarity scores and tamper detection results

### Data Integrity

- All saved files must include checksums for integrity verification
- Model files must preserve PyTorch state dictionary structure
- Watermark arrays must maintain exact shape matching classifier weights

## Constraints and Limitations

### Technical Constraints

1. Watermarking is limited to EfficientNet-B0 architecture and CIFAR-10 dataset
2. Watermark embedding modifies only the final classifier layer weights
3. Detection requires access to the original watermark array
4. AWOA optimization is computationally expensive and requires GPU acceleration
5. Flask API is designed for development/research use, not production-scale deployment

### Performance Constraints

1. Watermark embedding strength (alpha) must balance robustness and accuracy preservation
2. Detection threshold must be calibrated to minimize false positives and false negatives
3. Tamper detection accuracy depends on the severity of modifications
4. System cannot detect watermarks if classifier layer is completely replaced

### Security Constraints

1. HMAC secret keys must be stored securely and never transmitted over networks
2. System does not provide legal enforcement mechanisms for ownership disputes
3. Watermark security relies on keeping the original watermark array confidential
4. System cannot prevent determined attackers with full model access from removing watermarks

## Assumptions

1. Users have legitimate ownership rights to models they watermark
2. CIFAR-10 dataset is representative of the target application domain
3. EfficientNet-B0 architecture is suitable for the classification task
4. Attackers do not have access to the original watermark array
5. HMAC secret keys are generated securely and stored safely
6. Users will validate embedding strength before deploying watermarked models
7. The system operates in a trusted environment with secure file storage
8. Network communication between frontend and backend occurs over secure channels
9. Users understand basic deep learning concepts and model evaluation metrics

## Acceptance Criteria

### System-Level Acceptance Criteria

1. **Watermark Embedding Success:** System successfully embeds watermarks in EfficientNet-B0 models with accuracy degradation less than 2%
2. **Watermark Detection Success:** System detects embedded watermarks with cosine similarity above 0.95 for unmodified models
3. **Ownership Verification:** System generates HMAC-authenticated ownership certificates that can be cryptographically verified
4. **Tamper Detection:** System detects pruning, noise injection, and fine-tuning attacks with confidence scores
5. **Performance:** Embedding completes in under 5 seconds, detection in under 3 seconds on recommended hardware
6. **API Functionality:** All Flask API endpoints return correct responses for valid inputs and appropriate errors for invalid inputs
7. **Frontend Usability:** Web interface allows users to complete all operations without command-line access
8. **AWOA Optimization:** AWOA produces hyperparameters that achieve at least 85% test accuracy on CIFAR-10
9. **Data Persistence:** All models, watermarks, and certificates are saved correctly and can be loaded without errors
10. **Documentation:** System includes complete user documentation with examples for all operations

### Validation Criteria

1. Unit tests achieve at least 80% code coverage for core watermarking functions
2. Integration tests verify end-to-end workflows from training through detection
3. Security tests confirm HMAC authentication prevents certificate forgery
4. Performance tests confirm latency requirements under normal load
5. Tamper detection tests validate detection of known attack types
6. Reproducibility tests confirm identical results with fixed random seeds
