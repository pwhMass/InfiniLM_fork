use hyper::Method;
use openai_struct::{
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, CreateChatCompletionResponse,
    CreateChatCompletionResponseChoices, CreateChatCompletionStreamResponse,
    CreateChatCompletionStreamResponseChoices, FinishReason, Model,
};
use serde::Serialize;

const CHAT_COMPLETION_OBJECT: &str = "chat.completion.chunk";
pub(crate) const GET_MODELS: (&Method, &str) = (&Method::GET, "/models");
pub(crate) const POST_CHAT_COMPLETIONS: (&Method, &str) = (&Method::POST, "/chat/completions");

pub(crate) fn create_models(models: impl IntoIterator<Item = String>) -> impl Serialize {
    #[derive(Serialize)]
    struct Response {
        object: &'static str,
        data: Vec<Model>,
    }

    Response {
        object: "list",
        data: models
            .into_iter()
            .map(|id| Model {
                id,
                object: "model".into(),
                owned_by: "QYLab".into(),
                created: 0,
            })
            .collect(),
    }
}

pub(crate) fn chat_completion_response(
    id: usize,
    created: i32,
    model: String,
    think: Option<String>,
    answer: Option<String>,
    finish_reason: Option<FinishReason>,
) -> CreateChatCompletionResponse {
    let choices = vec![CreateChatCompletionResponseChoices {
        message: ChatCompletionResponseMessage {
            content: answer.unwrap(),
            reasoning_content: think,
            ..Default::default()
        },
        finish_reason,
        ..Default::default()
    }];
    CreateChatCompletionResponse {
        id: format!("InfiniLM-Service-chatcmpl-{id:#08x}"),
        object: CHAT_COMPLETION_OBJECT.to_string(),
        model,
        choices,
        created,
        ..Default::default()
    }
}

pub(crate) fn chat_completion_response_stream(
    id: usize,
    created: i32,
    model: String,
    think: Option<String>,
    answer: Option<String>,
    finish_reason: Option<FinishReason>,
) -> CreateChatCompletionStreamResponse {
    let choices = vec![CreateChatCompletionStreamResponseChoices {
        delta: ChatCompletionStreamResponseDelta {
            reasoning_content: think,
            content: answer,
            ..Default::default()
        },
        finish_reason,
        ..Default::default()
    }];
    CreateChatCompletionStreamResponse {
        id: format!("InfiniLM-Service-chatcmpl-{id:#08x}"),
        object: CHAT_COMPLETION_OBJECT.to_string(),
        created,
        model,
        choices,
        ..Default::default()
    }
}
