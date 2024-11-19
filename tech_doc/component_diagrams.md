# Curator Component Interaction Documentation

## 1. High-Level System Architecture
```mermaid
graph TB
    Client[Client Application] --> Prompter
    Prompter --> RequestProcessor
    Prompter --> Dataset
    RequestProcessor --> LLMProvider[LLM Provider]
    Dataset --> Storage[File Storage]
    RequestProcessor --> Storage
    
    subgraph Core Components
        Prompter[Prompter]
        RequestProcessor[Request Processor]
        Dataset[Dataset Manager]
    end
```

**Description:**
The high-level architecture shows the main components of Curator:
- **Client Application**: Entry point for using Curator
- **Prompter**: Central orchestrator managing prompt formatting and processing
- **Request Processor**: Handles API interactions with LLM providers
- **Dataset Manager**: Manages data storage and retrieval
- **Storage**: Persistent storage for requests, responses, and processed data

## 2. Initialization and Configuration Flow
```mermaid
sequenceDiagram
    participant Client
    participant Prompter
    participant PromptFormatter
    participant RequestProcessor
    participant MetadataDB

    Client->>Prompter: Initialize(model_name, prompt_func, parse_func)
    Prompter->>PromptFormatter: Create formatter
    Prompter->>RequestProcessor: Initialize processor
    Prompter->>MetadataDB: Store configuration
    MetadataDB-->>Prompter: Configuration stored
    Prompter-->>Client: Ready for processing
```

**Description:**
The initialization process involves:
1. Client configures Prompter with model and functions
2. Prompter creates PromptFormatter for structured prompts
3. RequestProcessor is initialized with rate limits
4. Configuration is stored in MetadataDB for tracking

## 3. Data Processing Pipeline
```mermaid
graph LR
    Input[Input Dataset] -->|Load| Prompter
    Prompter -->|Format| GenericRequest
    GenericRequest -->|Process| RequestProcessor
    RequestProcessor -->|Call API| LLMProvider
    LLMProvider -->|Response| GenericResponse
    GenericResponse -->|Parse| Dataset
    Dataset -->|Save| Storage

    subgraph Processing Pipeline
        GenericRequest
        RequestProcessor
        GenericResponse
    end
```

**Description:**
The data processing pipeline shows how data flows through the system:
1. Input data is loaded into Prompter
2. Prompter formats data into GenericRequests
3. RequestProcessor handles API communication
4. LLM Provider processes requests
5. Responses are converted to GenericResponses
6. Responses are parsed and validated
7. Results are saved to persistent storage

## 4. Batch Processing Architecture
```mermaid
graph TB
    Dataset -->|1| BatchProcessor
    BatchProcessor -->|2| RequestQueue[Request Queue]
    
    subgraph Workers
        RequestQueue -->|3| Worker1[Worker 1]
        RequestQueue -->|3| Worker2[Worker 2]
        RequestQueue -->|3| WorkerN[Worker N]
    end
    
    subgraph Rate Limiting
        Worker1 --> RateLimiter
        Worker2 --> RateLimiter
        WorkerN --> RateLimiter
    end
    
    RateLimiter -->|4| LLMProvider
    LLMProvider -->|5| ResponseCollector
    ResponseCollector -->|6| Storage
```

**Description:**
The batch processing system:
1. Splits dataset into batches
2. Queues requests for processing
3. Distributes work across multiple workers
4. Enforces rate limits
5. Collects and validates responses
6. Stores results persistently

## 5. Online Processing Flow
```mermaid
sequenceDiagram
    participant Client
    participant OnlineProcessor
    participant RateLimiter
    participant LLMProvider
    participant ErrorHandler

    Client->>OnlineProcessor: Submit request
    OnlineProcessor->>RateLimiter: Check capacity
    
    alt Has capacity
        RateLimiter->>OnlineProcessor: Allow request
        OnlineProcessor->>LLMProvider: Make API call
        
        alt Success
            LLMProvider-->>OnlineProcessor: Return response
            OnlineProcessor-->>Client: Return result
        else Error
            LLMProvider-->>ErrorHandler: Handle error
            ErrorHandler->>OnlineProcessor: Retry or fail
        end
        
    else No capacity
        RateLimiter->>OnlineProcessor: Throttle
        OnlineProcessor->>OnlineProcessor: Wait
    end
```

**Description:**
Online processing handles real-time requests:
- Checks rate limits before processing
- Makes API calls when capacity is available
- Handles errors and retries
- Implements backoff strategies
- Returns results immediately

## 6. Storage Architecture
```mermaid
graph TD
    RequestProcessor -->|Requests| RequestsJSONL[requests_*.jsonl]
    RequestProcessor -->|Responses| ResponsesJSONL[responses_*.jsonl]
    
    subgraph Persistent Storage
        RequestsJSONL --> ArrowDataset[dataset.arrow]
        ResponsesJSONL --> ArrowDataset
        ArrowDataset --> HuggingFace[HuggingFace Dataset]
    end
    
    subgraph Metadata Storage
        MetadataDB[(SQLite DB)]
        MetadataDB -->|Run History| RunTracking[Run Tracking]
        MetadataDB -->|Dataset Hash| Reproducibility
    end
```

**Description:**
The storage system provides:
- JSONL files for raw requests/responses (split by batch)
- Arrow datasets for efficient data processing
- SQLite database for run metadata and reproducibility
- HuggingFace dataset integration
- Run history tracking

## 7. Error Handling and Recovery
```mermaid
stateDiagram-v2
    [*] --> RequestCreation: Create Request
    
    RequestCreation --> RateCheck: Check Rate Limits
    RateCheck --> APICall: Capacity Available
    RateCheck --> Throttled: No Capacity
    
    Throttled --> CoolDown: Wait
    CoolDown --> RateCheck: Retry
    
    APICall --> Success: 200 OK
    APICall --> RateLimit: 429 Rate Limited
    APICall --> APIError: Other Error
    
    RateLimit --> ExponentialBackoff: Wait 15s
    ExponentialBackoff --> RateCheck: Retry
    
    APIError --> RetryCheck: Check Attempts Left
    RetryCheck --> APICall: Retry Available
    RetryCheck --> Failed: No Retries Left
    
    Success --> ResponseValidation: Parse Response
    ResponseValidation --> [*]: Valid
    ResponseValidation --> Failed: Invalid
    
    Failed --> ErrorLogging: Log Error
    ErrorLogging --> [*]: Complete
```

**Description:**
The error handling system implements:
- Pre-request rate limit checking
- Automatic throttling when near limits
- Response validation and parsing
- Multiple retry attempts (configurable)
- Exponential backoff for rate limits
- Detailed error logging
- Batch and request tracking

## 8. Component Interaction Details

### Prompter
- Manages prompt formatting and validation
- Handles request routing (batch/online)
- Integrates with metadata storage
- Manages caching and state

### Request Processor
- Implements provider-specific API calls
- Handles rate limiting and quotas
- Manages request batching
- Implements retry logic
- Handles response validation

### Dataset Manager
- Provides dataset iteration
- Handles format conversion
- Manages persistent storage
- Implements caching
- Supports HuggingFace integration

### Metadata Database
- Tracks run history
- Stores configurations
- Manages cache keys
- Enables reproducibility
- Provides analytics data

## 9. Integration Points
```mermaid
graph TB
    Curator[Curator Library] --> External[External Systems]
    
    subgraph External Systems
        HuggingFace[HuggingFace]
        OpenAI[OpenAI API]
        Anthropic[Anthropic API]
        Custom[Custom LLMs]
    end
    
    subgraph Integration Layer
        Adapters[Provider Adapters]
        DataConverters[Data Converters]
        APIClients[API Clients]
    end
    
    Curator --> Adapters
    Adapters --> External
```

**Description:**
Integration capabilities include:
- Multiple LLM provider support
- Dataset format conversion
- Custom provider integration
- External tool integration
- API compatibility layers