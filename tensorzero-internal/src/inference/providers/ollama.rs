use std::sync::OnceLock;

use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse};
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, ContentBlockOutput, FinishReason, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseArgs, ProviderInferenceResponseChunk, ProviderInferenceResponseStreamInner,
    RequestMessage, Role, TextChunk, Usage,
};
use crate::model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider};

use super::helpers::inject_extra_body;

lazy_static! {
    static ref OLLAMA_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("http://localhost:11434/api").expect("Failed to parse OLLAMA_DEFAULT_BASE_URL")
    };
}

fn default_api_key_location() -> CredentialLocation {
    // Ollama doesn't require an API key, included for consistency reasons
    CredentialLocation::Env("OLLAMA_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "Ollama";
const PROVIDER_TYPE: &str = "ollama";

#[derive(Debug)]
pub struct OllamaProvider {
    model_name: String,
    api_base: Option<Url>,
    credentials: OllamaCredentials,
}

static DEFAULT_CREDENTIALS: OnceLock<OllamaCredentials> = OnceLock::new();

impl OllamaProvider {
    pub fn new(
        model_name: String,
        api_base: Option<Url>,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credentials = build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        Ok(OllamaProvider {
            model_name,
            api_base,
            credentials,
        })
    }
}

#[derive(Clone, Debug)]
pub enum OllamaCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for OllamaCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(OllamaCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(OllamaCredentials::Dynamic(key_name)),
            Credential::None => Ok(OllamaCredentials::None),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(OllamaCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Ollama provider".to_string(),
            })),
        }
    }
}

impl OllamaCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            OllamaCredentials::Static(api_key) => Ok(Some(api_key)),
            OllamaCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                }))
                .transpose()
            }
            OllamaCredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for OllamaProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let mut request_body = serde_json::to_value(OllamaRequest::new(&self.model_name, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error serializing Ollama request: {e}"),
                })
            })?;
        inject_extra_body(request.extra_body, model_provider, &mut request_body)?;
        let request_url = get_generate_url(self.api_base.as_ref().unwrap_or(&OLLAMA_DEFAULT_BASE_URL))?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json");
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to Ollama: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            Ok(OllamaResponseWithMetadata {
                response,
                latency,
                raw_response,
                request: request_body,
                generic_request: request,
            }
            .try_into()?)
        } else {
            Err(handle_ollama_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing error response: {e}"),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                        raw_response: None,
                    })
                })?,
            ))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let mut request_body = serde_json::to_value(OllamaRequest::new(&self.model_name, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error serializing Ollama request: {e}"),
                })
            })?;
        inject_extra_body(request.extra_body, model_provider, &mut request_body)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let request_url = get_generate_url(self.api_base.as_ref().unwrap_or(&OLLAMA_DEFAULT_BASE_URL))?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json");
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let event_source = request_builder
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to Ollama: {e}"),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

        let stream = stream_ollama(event_source, start_time).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

fn get_generate_url(base_url: &Url) -> Result<Url, Error> {
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    url.join("generate").map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}

fn handle_ollama_error(
    response_code: reqwest::StatusCode,
    response_body: &str,
) -> Error {
    match response_code {
        reqwest::StatusCode::BAD_REQUEST
        | reqwest::StatusCode::UNAUTHORIZED
        | reqwest::StatusCode::FORBIDDEN
        | reqwest::StatusCode::TOO_MANY_REQUESTS => ErrorDetails::InferenceClient {
            status_code: Some(response_code),
            message: response_body.to_string(),
            raw_request: None,
            raw_response: None,
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            raw_request: None,
            raw_response: None,
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into(),
    }
}

// Ollama request format
#[derive(Debug, Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>, // max_tokens equivalent
}

impl<'a> OllamaRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<OllamaRequest<'a>, Error> {
        // Combine messages into a single prompt string
        let prompt = format_messages_as_prompt(&request.messages)?;
        
        let options = if request.temperature.is_some() 
            || request.top_p.is_some() 
            || request.frequency_penalty.is_some() 
            || request.presence_penalty.is_some() 
            || request.seed.is_some() 
            || request.max_tokens.is_some() {
            Some(OllamaOptions {
                temperature: request.temperature,
                top_p: request.top_p,
                frequency_penalty: request.frequency_penalty,
                presence_penalty: request.presence_penalty,
                seed: request.seed,
                num_predict: request.max_tokens,
            })
        } else {
            None
        };

        Ok(OllamaRequest {
            model,
            prompt,
            system: request.system.clone(),
            template: None,
            context: None,
            format: match request.json_mode {
                ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => Some("json".to_string()),
                ModelInferenceRequestJsonMode::Off => None,
            },
            raw: None,
            options,
            stream: request.stream,
        })
    }
}

fn format_messages_as_prompt(messages: &[RequestMessage]) -> Result<String, Error> {
    let mut prompt = String::new();
    
    for message in messages {
        let role_prefix = match message.role {
            Role::User => "User: ",
            Role::Assistant => "Assistant: ",
        };
        
        prompt.push_str(role_prefix);
        
        for block in &message.content {
            match block {
                ContentBlock::Text(text) => {
                    prompt.push_str(&text.text);
                    prompt.push_str("\n");
                },
                ContentBlock::ToolCall(_) => {
                    return Err(Error::new(ErrorDetails::InvalidMessage {
                        message: "Tool calls not supported in Ollama".to_string(),
                    }));
                },
                ContentBlock::ToolResult(_) => {
                    return Err(Error::new(ErrorDetails::InvalidMessage {
                        message: "Tool results not supported in Ollama".to_string(),
                    }));
                },
                ContentBlock::Image(_) => {
                    return Err(Error::new(ErrorDetails::InvalidMessage {
                        message: "Images not supported in Ollama".to_string(),
                    }));
                },
                ContentBlock::Thought(_) => {
                    // Skip thought blocks as they're internal
                    continue;
                },
                ContentBlock::Unknown { .. } => {
                    return Err(Error::new(ErrorDetails::InvalidMessage {
                        message: "Unknown content block not supported in Ollama".to_string(),
                    }));
                },
            }
        }
        
        prompt.push_str("\n");
    }
    
    // Add a final "Assistant: " prompt to indicate it's the model's turn
    prompt.push_str("Assistant: ");
    
    Ok(prompt)
}

// Ollama response format
#[derive(Debug, Deserialize, Clone)]
struct OllamaResponse {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,
    eval_duration: Option<u64>,
}

struct OllamaResponseWithMetadata<'a> {
    response: OllamaResponse,
    raw_response: String,
    latency: Latency,
    request: serde_json::Value,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<OllamaResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: OllamaResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OllamaResponseWithMetadata {
            response,
            latency,
            request: request_body,
            generic_request,
            raw_response,
        } = value;

        let finish_reason = if response.done {
            Some(FinishReason::Stop)
        } else {
            None
        };

        let content = vec![ContentBlockOutput::Text(crate::inference::types::Text {
            text: response.response
        })];
        
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request body as JSON: {e}"),
            })
        })?;
        
        // Calculate usage based on available metrics
        let usage = Usage {
            input_tokens: response.prompt_eval_count.unwrap_or(0),
            output_tokens: response.eval_count.unwrap_or(0),
        };

        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response: raw_response.clone(),
                usage,
                latency,
                finish_reason,
            },
        ))
    }
}

fn stream_ollama(
    mut event_source: EventSource,
    start_time: Instant,
) -> ProviderInferenceResponseStreamInner {
    Box::pin(async_stream::stream! {
        let mut prompt_eval_count = 0;
        let mut eval_count = 0;
        
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(PROVIDER_TYPE.to_string(), e).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<OllamaResponse, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: None,
                                raw_response: None,
                            }));
                        
                        let latency = start_time.elapsed();
                        
                        if let Ok(data) = data {
                            // Update token counts
                            if let Some(p_count) = data.prompt_eval_count {
                                prompt_eval_count = p_count;
                            }
                            if let Some(e_count) = data.eval_count {
                                eval_count = e_count;
                            }
                            
                            let finish_reason = if data.done {
                                Some(FinishReason::Stop)
                            } else {
                                None
                            };
                            
                            let content = vec![ContentBlockChunk::Text(TextChunk {
                                text: data.response,
                                id: "0".to_string(),
                            })];
                            
                            let usage = if data.done {
                                Some(Usage {
                                    input_tokens: prompt_eval_count,
                                    output_tokens: eval_count,
                                })
                            } else {
                                None
                            };
                            
                            let raw_response = message.data;
                            
                            yield Ok(ProviderInferenceResponseChunk::new(
                                content,
                                usage,
                                raw_response,
                                latency,
                                finish_reason,
                            ));
                            
                            if data.done {
                                break;
                            }
                        } else {
                            yield Err(data.unwrap_err());
                        }
                    },
                },
            }
        }

        event_source.close();
    })
}

async fn convert_stream_error(provider_type: String, e: reqwest_eventsource::Error) -> Error {
    let message = e.to_string();
    let mut raw_response = None;
    if let reqwest_eventsource::Error::InvalidStatusCode(_, resp) = e {
        raw_response = resp.text().await.ok();
    }
    ErrorDetails::InferenceServer {
        message,
        raw_request: None,
        raw_response,
        provider_type,
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use uuid::Uuid;
    use crate::inference::types::{FunctionType, Role};

    #[test]
    fn test_ollama_request_new() {
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![
                RequestMessage {
                    role: Role::User,
                    content: vec!["Hello".to_string().into()],
                },
                RequestMessage {
                    role: Role::Assistant,
                    content: vec!["Hi there!".to_string().into()],
                },
            ],
            system: Some("You are a helpful assistant".to_string()),
            temperature: Some(0.7),
            max_tokens: Some(100),
            seed: Some(42),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: None,
            ..Default::default()
        };

        let ollama_request = OllamaRequest::new("llama3", &request).unwrap();

        assert_eq!(ollama_request.model, "llama3");
        assert_eq!(ollama_request.prompt, "User: Hello\n\nAssistant: Hi there!\n\nAssistant: ");
        assert_eq!(ollama_request.system, Some("You are a helpful assistant".to_string()));
        assert!(ollama_request.stream);
        assert!(ollama_request.format.is_none());
        
        let options = ollama_request.options.unwrap();
        assert_eq!(options.temperature, Some(0.7));
        assert_eq!(options.num_predict, Some(100));
        assert_eq!(options.seed, Some(42));
        assert_eq!(options.top_p, Some(0.9));
        assert_eq!(options.presence_penalty, Some(0.1));
        assert_eq!(options.frequency_penalty, Some(0.2));
    }

    #[test]
    fn test_ollama_response_with_metadata_try_into() {
        let ollama_response = OllamaResponse {
            model: "llama3".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            response: "Hello, world!".to_string(),
            done: true,
            total_duration: Some(1000),
            load_duration: Some(200),
            prompt_eval_count: Some(10),
            prompt_eval_duration: Some(300),
            eval_count: Some(20),
            eval_duration: Some(500),
        };

        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: None,
            ..Default::default()
        };

        let ollama_response_with_metadata = OllamaResponseWithMetadata {
            response: ollama_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(1),
            },
            request: serde_json::to_value(OllamaRequest::new("llama3", &generic_request).unwrap()).unwrap(),
            generic_request: &generic_request,
        };

        let inference_response: ProviderInferenceResponse = ollama_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 1);
        assert_eq!(inference_response.output[0], "Hello, world!".to_string().into());
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(inference_response.raw_response, "test_response");
        assert_eq!(inference_response.usage.input_tokens, 10);
        assert_eq!(inference_response.usage.output_tokens, 20);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_secs(1)
            }
        );
    }

    #[test]
    fn test_format_messages_as_prompt() {
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["Hi there!".to_string().into()],
            },
            RequestMessage {
                role: Role::User,
                content: vec!["How are you?".to_string().into()],
            },
        ];

        let prompt = format_messages_as_prompt(&messages).unwrap();
        assert_eq!(prompt, "User: Hello\n\nAssistant: Hi there!\n\nUser: How are you?\n\nAssistant: ");
    }

    #[test]
    fn test_get_generate_url() {
        let base_url = Url::parse("http://localhost:11434/api").unwrap();
        let generate_url = get_generate_url(&base_url).unwrap();
        assert_eq!(generate_url.as_str(), "http://localhost:11434/api/generate");

        let base_url_no_trailing_slash = Url::parse("http://localhost:11434/api").unwrap();
        let generate_url = get_generate_url(&base_url_no_trailing_slash).unwrap();
        assert_eq!(generate_url.as_str(), "http://localhost:11434/api/generate");
    }

    #[test]
    fn test_credential_to_ollama_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: OllamaCredentials = OllamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OllamaCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = OllamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OllamaCredentials::Dynamic(_)));

        // Test None credential
        let generic = Credential::None;
        let creds = OllamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OllamaCredentials::None));

        // Test Missing credential (in test mode)
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            let generic = Credential::Missing;
            let creds = OllamaCredentials::try_from(generic).unwrap();
            assert!(matches!(creds, OllamaCredentials::None));
        }

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = OllamaCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
}