// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package gemini implements the [model.LLM] interface for Gemini models.
package gemini

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"runtime"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/llminternal/converters"
	"google.golang.org/adk/internal/version"
	"google.golang.org/adk/model"
)

// TODO: test coverage
type geminiModel struct {
	client             *genai.Client
	name               string
	versionHeaderValue string
}

// NewModel returns [model.LLM], backed by the Gemini API.
//
// It uses the provided context and configuration to initialize the underlying
// [genai.Client]. The modelName specifies which Gemini model to target
// (e.g., "gemini-2.5-flash").
//
// An error is returned if the [genai.Client] fails to initialize.
func NewModel(ctx context.Context, modelName string, cfg *genai.ClientConfig) (model.LLM, error) {
	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		return nil, err
	}

	// Create header value once, when the model is created
	headerValue := fmt.Sprintf("google-adk/%s gl-go/%s", version.Version,
		strings.TrimPrefix(runtime.Version(), "go"))

	return &geminiModel{
		name:               modelName,
		client:             client,
		versionHeaderValue: headerValue,
	}, nil
}

func (m *geminiModel) Name() string {
	return m.name
}

// GenerateContent calls the underlying model.
func (m *geminiModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.HTTPOptions == nil {
		req.Config.HTTPOptions = &genai.HTTPOptions{}
	}
	if req.Config.HTTPOptions.Headers == nil {
		req.Config.HTTPOptions.Headers = make(http.Header)
	}
	m.addHeaders(req.Config.HTTPOptions.Headers)

	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// addHeaders sets the x-goog-api-client and user-agent headers
func (m *geminiModel) addHeaders(headers http.Header) {
	headers.Set("x-goog-api-client", m.versionHeaderValue)
	headers.Set("user-agent", m.versionHeaderValue)
}

// generate calls the model synchronously returning result from the first candidate.
func (m *geminiModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	resp, err := m.client.Models.GenerateContent(ctx, m.name, req.Contents, req.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}
	if len(resp.Candidates) == 0 {
		// shouldn't happen?
		return nil, fmt.Errorf("empty response")
	}
	return converters.Genai2LLMResponse(resp), nil
}

// generateStream returns a stream of responses from the model.
func (m *geminiModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	aggregator := llminternal.NewStreamingResponseAggregator()

	return func(yield func(*model.LLMResponse, error) bool) {
		for resp, err := range m.client.Models.GenerateContentStream(ctx, m.name, req.Contents, req.Config) {
			if err != nil {
				yield(nil, err)
				return
			}
			for llmResponse, err := range aggregator.ProcessResponse(ctx, resp) {
				if !yield(llmResponse, err) {
					return // Consumer stopped
				}
			}
		}
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

// maybeAppendUserContent appends a user content, so that model can continue to output.
func (m *geminiModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}

// Connect establishes a bidirectional streaming connection to the model.
func (m *geminiModel) Connect(ctx context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
	// Map LLMRequest config to LiveConnectConfig
	var modalities []genai.Modality
	if req.Config.ResponseModalities != nil {
		modalities = make([]genai.Modality, len(req.Config.ResponseModalities))
		for i, m := range req.Config.ResponseModalities {
			modalities[i] = genai.Modality(m)
		}
	}

	config := &genai.LiveConnectConfig{
		// HTTPOptions:        req.Config.HTTPOptions,
		ResponseModalities: modalities,
		// Temperature:        req.Config.Temperature,
	}

	if req.LiveConnectConfig != nil {
		slog.Info("Gemini.Connect.config", "req.LiveConnectConfig", req.LiveConnectConfig)
		config = req.LiveConnectConfig
	}

	// System instruction handling
	if req.Config != nil && req.Config.SystemInstruction != nil {
		config.SystemInstruction = req.Config.SystemInstruction
	}

	slog.Info("Gemini.Connect.config", "config", config)
	slog.Info("Gemini.Connect.config", "model", m.name, "config", config)

	session, err := m.client.Live.Connect(ctx, m.name, config)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to live model: %w", err)
	}

	return &liveConnection{
		session: session,
	}, nil
}

type liveConnection struct {
	session *genai.Session
}

func (c *liveConnection) Send(req *model.LiveRequest) error {
	if req.Close {
		return c.Close()
	}
	if req.Content != nil {
		// Wrap single content in slice as LiveClientContentInput expects turns
		turnComplete := true
		return c.session.SendClientContent(genai.LiveClientContentInput{
			Turns:        []*genai.Content{req.Content},
			TurnComplete: &turnComplete,
		})
	}
	if req.RealtimeInput != nil {
		return c.session.SendRealtimeInput(*req.RealtimeInput)
	}
	if req.ToolResponse != nil {
		return c.session.SendToolResponse(*req.ToolResponse)
	}
	return nil
}

func (c *liveConnection) Receive() (*model.LLMResponse, error) {
	msg, err := c.session.Receive()
	if err != nil {
		slog.Error("Gemini.Receive.error", "error", err)
		return nil, err
	}

	slog.Info("Gemini.Receive.msg", "msg", msg)

	resp := &model.LLMResponse{}

	if msg.ServerContent != nil {
		// Map ServerContent to model.LLMResponse
		if msg.ServerContent.ModelTurn != nil {
			resp.Content = msg.ServerContent.ModelTurn
		}
		resp.TurnComplete = msg.ServerContent.TurnComplete
		resp.Interrupted = msg.ServerContent.Interrupted
	}

	if msg.ToolCall != nil {
		// Map ToolCall to model.LLMResponse content parts
		parts := make([]*genai.Part, 0)
		for _, fc := range msg.ToolCall.FunctionCalls {
			parts = append(parts, &genai.Part{
				FunctionCall: fc,
			})
		}
		if resp.Content == nil {
			resp.Content = &genai.Content{Role: "model"}
		}
		resp.Content.Parts = append(resp.Content.Parts, parts...)
	}

	if msg.ToolCallCancellation != nil {
		// TODO: Handle cancellation? Maybe just return empty response or specific error?
	}

	return resp, nil
}

func (c *liveConnection) Close() error {
	return c.session.Close()
}
