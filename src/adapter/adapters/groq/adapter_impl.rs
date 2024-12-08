use crate::adapter::openai::OpenAIAdapter;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{ChatOptionsSet, ChatRequest, ChatResponse, ChatStreamResponse};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{ClientConfig, ModelIden};
use crate::{Result, ServiceTarget};
use reqwest::RequestBuilder;

pub struct GroqAdapter;

pub(in crate::adapter) const MODELS: &[&str] = &[
	"llama-3.2-90b-vision-preview",
	"llama-3.2-11b-vision-preview",
	"llama-3.2-3b-preview",
	"llama-3.2-1b-preview",
	"llama-3.1-405b-reasoning",
	"llama-3.1-70b-versatile",
	"llama-3.1-8b-instant",
	"mixtral-8x7b-32768",
	"gemma-7b-it",
	"gemma2-9b-it",
	"llama3-groq-70b-8192-tool-use-preview",
	"llama3-groq-8b-8192-tool-use-preview",
	"llama3-8b-8192",
	"llama3-70b-8192",
	// "whisper-large-v3", // This is not a chat completion model
];

// The Groq API adapter is modeled after the OpenAI adapter, as the Groq API is compatible with the OpenAI API.
impl Adapter for GroqAdapter {
	fn default_endpoint(kind: AdapterKind) -> Endpoint {
		const BASE_URL: &str = "https://api.groq.com/openai/v1/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth(kind: AdapterKind) -> AuthData {
		AuthData::from_env("GROQ_API_KEY")
	}

	async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
		Ok(MODELS.iter().map(|s| s.to_string()).collect())
	}

	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> String {
		OpenAIAdapter::util_get_service_url(model, service_type, endpoint)
	}

	fn to_web_request_data(
		target: ServiceTarget,
		client_config: &ClientConfig,
		service_type: ServiceType,
		chat_req: ChatRequest,
		chat_options: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		OpenAIAdapter::util_to_web_request_data(target, service_type, chat_req, chat_options)
	}

	fn to_chat_response(model_iden: ModelIden, web_response: WebResponse) -> Result<ChatResponse> {
		OpenAIAdapter::to_chat_response(model_iden, web_response)
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		OpenAIAdapter::to_chat_stream(model_iden, reqwest_builder, options_set)
	}
}
