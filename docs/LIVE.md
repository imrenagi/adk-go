# Gemini Live API in adk-go

This document describes how to use the Gemini Live API support in the `adk-go` library for real-time, bidirectional streaming interactions.

## Overview

The Live API enables real-time interaction with Gemini models using WebSockets. Unlike standard unary or single-stream requests, Live mode allow for:

- **Bidirectional communication**: Send and receive messages simultaneously.
- **Low latency**: Reduced overhead for continuous interactions.
- **Advanced modalities**: Support for audio input/output and real-time tool use.

## Key Components

### 1. `StreamingModeBidi`

To enable live streaming, you must set the `StreamingMode` to `StreamingModeBidi` in the `agent.RunConfig`.

```go
runConfig := agent.RunConfig{
    StreamingMode: agent.StreamingModeBidi,
    // ... other configs
}
```

### 2. `LiveRequestQueue`

The `LiveRequestQueue` is a thread-safe queue used to send client-side messages (text, audio, or tool responses) to the model during an active live session.

```go
queue := agent.NewLiveRequestQueue()
runConfig.LiveRequestQueue = queue

// Send text content to the model
queue.SendContent(genai.NewContentFromText("Hello Gemini", genai.RoleUser))

// Close the queue when the session is finished
defer queue.Close()
```

### 3. `LiveConnectConfig`

The `LiveConnectConfig` (from `google.golang.org/genai`) allows you to configure the live session, including:

- **ResponseModalities**: Specify if the model should return text, audio, or both.
- **SpeechConfig**: Configure the voice and language for audio output.
- **SystemInstruction**: Provide system-level guidelines for the live session.

```go
runConfig.LiveConnectConfig = &genai.LiveConnectConfig{
    ResponseModalities: []genai.Modality{genai.ModalityAudio, genai.ModalityText},
    SpeechConfig: &genai.SpeechConfig{
        VoiceConfig: &genai.VoiceConfig{
            PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
                VoiceName: "Aoide",
            },
        },
    },
}
```

## Usage Example

Here is a simplified example of how to start a live session using the `Runner`. For a complete implementation, see `examples/live/main.go`.

```go
// 1. Setup Model and Agent
model, _ := gemini.NewModel(ctx, "gemini-2.5-flash-native-audio-preview-12-2025", clientConfig)
a, _ := llmagent.New(llmagent.Config{
    Name: "my_agent",
    Model: model,
    Instruction: "You are a helpful assistant.",
})

// 2. Setup Runner and Session
r, _ := runner.New(runner.Config{ Agent: a, SessionService: session.InMemoryService() })
sessionID := "my-session"

// 3. Configure Live Run
queue := agent.NewLiveRequestQueue()
runConfig := agent.RunConfig{
    StreamingMode:    agent.StreamingModeBidi,
    LiveRequestQueue: queue,
    LiveConnectConfig: &genai.LiveConnectConfig{
        ResponseModalities: []genai.Modality{genai.ModalityText},
    },
}

// 4. Run the Agent
userContent := genai.NewContentFromText("Start session", genai.RoleUser)
go func() {
    for ev, err := range r.Run(ctx, "user-id", sessionID, userContent, runConfig) {
        if err != nil {
            fmt.Printf("Error: %v\n", err)
            return
        }
        // Handle model response events
        if ev.LLMResponse.Content != nil {
            fmt.Printf("Agent: %s\n", ev.LLMResponse.Content.Parts[0].Text)
        }
    }
}()

// 5. Interact via the queue
queue.SendContent(genai.NewContentFromText("Tell me a joke", genai.RoleUser))
```

## Supported Models

Currently, Live API support is implemented for Gemini models in the `model/gemini` package. Ensure you use a model name that supports bidirectional streaming and native audio (if using audio features), such as `gemini-2.5-flash-native-audio-preview`.

## Troubleshooting

- **"Cannot extract voices from a non-audio request"**: Ensure you have configured `SpeechConfig` or included `AUDIO` in `ResponseModalities` if the model expects to generate speech.
- **WebSocket connection closed**: Check your network connectivity and API key permissions. Ensure the model supports `bidiGenerateContent`.
