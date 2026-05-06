//! Replay integration tests for the Gemini adapter.
//!
//! These tests use pre-recorded cassettes from `tests/data/yakbak/gemini/`
//! and assert that thinking content, tool calls, and usage flow through correctly.

mod support;

use genai::chat::*;
use serde_json::json;
use support::yakbak::replay_client;
use support::{TestResult, extract_stream_end};

#[tokio::test]
async fn test_yakbak_gemini_thinking_stream() -> TestResult<()> {
	let (client, _server) = replay_client("gemini", "thinking_stream").await?;

	let chat_req = ChatRequest::new(vec![
		ChatMessage::system("Answer in one sentence."),
		ChatMessage::user("Why is the sky blue?"),
	]);
	let options = ChatOptions::default()
		.with_reasoning_effort(ReasoningEffort::Low)
		.with_capture_content(true)
		.with_capture_reasoning_content(true)
		.with_capture_usage(true);

	let stream_res = client.exec_chat_stream("gemini-2.5-flash", chat_req, Some(&options)).await?;
	let extract = extract_stream_end(stream_res.stream).await?;

	// Exact text
	assert_eq!(
		extract.content.as_deref(),
		Some(
			"The sky is blue because the Earth's atmosphere, primarily nitrogen and oxygen molecules, scatters shorter, bluer wavelengths of sunlight more efficiently than longer wavelengths, dispersing blue light across the sky."
		),
	);

	// Reasoning should be substantial (1604 chars from recorded)
	let reasoning = extract.reasoning_content.as_deref().ok_or("Should have reasoning")?;
	assert_eq!(reasoning.len(), 1604, "reasoning length should be exactly 1604 chars");
	assert!(reasoning.starts_with("**Defining Atmospheric Color**"));

	// Exact usage
	let usage = extract.stream_end.captured_usage.as_ref().ok_or("Should have usage")?;
	assert_eq!(usage.prompt_tokens, Some(12));
	assert_eq!(usage.completion_tokens, Some(732));

	Ok(())
}

#[tokio::test]
async fn test_yakbak_gemini_tool_stream() -> TestResult<()> {
	let (client, _server) = replay_client("gemini", "tool_stream").await?;

	let chat_req = ChatRequest::new(vec![
		ChatMessage::system("You MUST use the get_weather tool to answer weather questions. Do not answer from prior knowledge."),
		ChatMessage::user("Use the get_weather tool to fetch the current weather for Paris, France in Celsius."),
	])
	.append_tool(Tool::new("get_weather").with_schema(json!({
		"type": "object",
		"properties": {
			"city": { "type": "string", "description": "The city name" },
			"country": { "type": "string", "description": "The country" },
			"unit": { "type": "string", "enum": ["C", "F"], "description": "Temperature unit" }
		},
		"required": ["city", "country", "unit"],
	})));

	let options = ChatOptions::default()
		.with_capture_content(true)
		.with_capture_tool_calls(true)
		.with_capture_usage(true);

	let stream_res = client.exec_chat_stream("gemini-2.5-flash", chat_req, Some(&options)).await?;
	let extract = extract_stream_end(stream_res.stream).await?;

	// -- Tool call streamed via ToolCallChunk
	assert_eq!(extract.tool_call_chunks.len(), 1, "expected one streamed tool call");
	let streamed_tc = &extract.tool_call_chunks[0];
	assert_eq!(streamed_tc.fn_name, "get_weather");
	assert_eq!(streamed_tc.fn_arguments["city"], "Paris");
	assert_eq!(streamed_tc.fn_arguments["country"], "France");
	assert_eq!(streamed_tc.fn_arguments["unit"], "C");

	// -- Tool call also captured at end
	let captured = extract
		.stream_end
		.captured_tool_calls()
		.ok_or("Should have captured tool calls")?;
	assert_eq!(captured.len(), 1);
	assert_eq!(captured[0].fn_name, "get_weather");

	// -- Stop reason set from finishReason
	assert_eq!(extract.stream_end.captured_stop_reason, Some(StopReason::Completed("STOP".into())));

	// -- Usage captured
	let usage = extract.stream_end.captured_usage.as_ref().ok_or("Should have usage")?;
	assert_eq!(usage.prompt_tokens, Some(102));
	// completion_tokens = candidatesTokenCount(25) + thoughtsTokenCount(84) = 109
	assert_eq!(usage.completion_tokens, Some(109));

	Ok(())
}
