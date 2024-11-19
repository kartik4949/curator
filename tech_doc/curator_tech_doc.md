# Curator Technical Documentation

## Overview

Curator is a Python library designed for large-scale synthetic data generation using Large Language Models (LLMs). It provides a robust framework for:
- Structured prompting of LLMs
- Efficient batch and online processing
- Response validation and parsing
- Dataset management and caching
- Error handling and rate limiting

### Key Features
- Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- Batch and online processing modes
- Built-in rate limiting and error handling
- Structured output validation using Pydantic
- Caching and resumption capabilities
- Dataset integration with HuggingFace

## Core Components

### 1. Prompter
The central orchestrator for LLM interactions (`prompter/prompter.py`).

#### Key Responsibilities
- Manages prompt formatting and response parsing
- Handles request routing (batch/online)
- Integrates with metadata storage
- Manages caching and state

#### Example Usage
```python
from bespokelabs import curator
from pydantic import BaseModel, Field
from typing import List

class Topics(BaseModel):
    topics_list: List[str] = Field(description="A list of topics.")

# Create a prompter for generating topics
topic_generator = curator.Prompter(
    prompt_func=lambda: "Generate 10 diverse topics for poems.",
    model_name="gpt-4",
    response_format=Topics,
    parse_func=lambda _, topics: [{"topic": t} for t in topics.topics_list],
)

# Generate topics
topics_dataset = topic_generator()
```

### 2. Dataset Management
Handles data processing and storage (`dataset.py`).

#### Features
- Dataset iteration and conversion
- HuggingFace dataset integration
- File-based storage (JSONL, Arrow)
- Efficient batch processing

#### Example
```python
# Chain prompters to build complex datasets
poems_dataset = poet_prompter(topics_dataset)
print(poems_dataset.to_pandas())
```

### 3. Request Processing
Manages LLM API interactions with two main modes:

#### Online Request Processor
(`request_processor/openai_online_request_processor.py`)
- Real-time API requests
- Dynamic rate limiting
- Parallel processing
- Automatic retries

#### Batch Request Processor
(`request_processor/openai_batch_request_processor.py`)
- Bulk request handling
- Optimized for large datasets
- State management
- Efficient resource usage

### 4. Metadata Management
SQLite-based tracking system (`db.py`):
- Run history tracking
- Configuration storage
- Cache management
- Run reproducibility

## Usage Examples

### 1. Basic Topic Generation
```python
# Example from examples/poem.py
from bespokelabs import curator
from pydantic import BaseModel, Field
from typing import List

class Topics(BaseModel):
    topics_list: List[str] = Field(description="A list of topics.")

topic_generator = curator.Prompter(
    prompt_func=lambda: "Generate 10 diverse topics.",
    model_name="gpt-4",
    response_format=Topics,
    parse_func=lambda _, topics: [{"topic": t} for t in topics.topics_list],
)

topics = topic_generator()
```

### 2. Chained Processing
```python
# Example from examples/camel.py
subject_prompter = curator.Prompter(
    prompt_func=lambda: "Generate subjects",
    response_format=Subjects,
)
subjects = subject_prompter()

qa_prompter = curator.Prompter(
    prompt_func=lambda subject: f"Generate QA for {subject}",
    response_format=QAs,
)
qa_dataset = qa_prompter(subjects)
```

### 3. Batch Processing
```python
# Example from examples/distill.py
distill_prompter = curator.Prompter(
    prompt_func=prompt_func,
    parse_func=parse_func,
    model_name="gpt-4",
    batch=True,
    batch_size=10,
)

distilled_dataset = distill_prompter(dataset)
```

## Developer Guide

### Key Concepts

1. **Prompt Function**
- Takes a dataset row as input
- Returns formatted prompt for LLM
- Can be synchronous or async
```python
def prompt_func(row: Dict[str, Any]) -> Union[str, Dict[str, str]]:
    return f"Process this: {row['data']}"
```

2. **Response Format**
- Pydantic model for structured output
- Validates LLM responses
- Ensures data consistency
```python
class ResponseFormat(BaseModel):
    field1: str
    field2: int
```

3. **Parse Function**
- Processes LLM response
- Maps to dataset structure
- Can generate multiple rows
```python
def parse_func(row: Dict[str, Any], response: Any) -> List[Dict[str, Any]]:
    return [{"processed": response.result}]
```

### Adding New Features

#### 1. New LLM Provider
1. Create new request processor:
```python
class NewProviderProcessor(BaseRequestProcessor):
    def get_rate_limits(self) -> dict:
        return {
            "max_requests_per_minute": 100,
            "max_tokens_per_minute": 10000
        }
```

2. Implement API-specific handling:
```python
def create_api_specific_request(
    self, 
    generic_request: GenericRequest
) -> dict:
    return {
        "model": self.model,
        "prompt": generic_request.messages[0]["content"],
        "temperature": self.temperature
    }
```

#### 2. Custom Processing Modes
1. Extend BaseRequestProcessor
2. Implement rate limiting
3. Add error handling
4. Manage state

### Best Practices

1. **Error Handling**
- Implement retries with exponential backoff
- Log errors comprehensively
- Provide meaningful error messages
```python
try:
    response = await self.make_request(request)
except RateLimitError:
    await self.handle_rate_limit()
except APIError as e:
    logger.error(f"API error: {e}")
```

2. **Rate Limiting**
- Respect provider limits
- Implement token counting
- Use cooldown periods
```python
if self._should_throttle():
    await self._wait_for_capacity()
```

3. **Testing**
- Unit test new components
- Test with small datasets
- Verify error paths
- Mock API responses

4. **Performance**
- Use batch processing for large datasets
- Implement efficient caching
- Monitor resource usage

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**
```python
# Adjust batch size
prompter = Prompter(
    batch=True,
    batch_size=5  # Reduce from default
)
```

2. **Memory Issues**
```python
# Use streaming for large datasets
dataset = Dataset.from_generator(generator)
```

3. **Response Parsing**
```python
# Validate response format
class ResponseFormat(BaseModel):
    field: str
    
    @validator('field')
    def validate_field(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v
```

## Future Development

1. **Provider Support**
- Additional LLM providers
- Custom API integration
- Local model support

2. **Features**
- Streaming responses
- Advanced caching
- Cost optimization

3. **Monitoring**
- Real-time metrics
- Cost tracking
- Performance analysis

## Contributing

1. Fork repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

Follow:
- Code style guide
- Documentation standards
- Test coverage requirements