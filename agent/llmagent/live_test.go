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

package llmagent_test

import (
	"context"
	"fmt"
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// MockLiveConnection simulates a live connection.
type MockLiveConnection struct {
	SendFunc    func(req *model.LiveRequest) error
	ReceiveFunc func() (*model.LLMResponse, error)
	CloseFunc   func() error
}

func (m *MockLiveConnection) Send(req *model.LiveRequest) error {
	if m.SendFunc != nil {
		return m.SendFunc(req)
	}
	return nil
}

func (m *MockLiveConnection) Receive() (*model.LLMResponse, error) {
	if m.ReceiveFunc != nil {
		return m.ReceiveFunc()
	}
	return nil, nil
}

func (m *MockLiveConnection) Close() error {
	if m.CloseFunc != nil {
		return m.CloseFunc()
	}
	return nil
}

// MockLiveLLM simulates an LLM that supports Connect.
type MockLiveLLM struct {
	ConnectFunc func(ctx context.Context, req *model.LLMRequest) (model.LiveConnection, error)
}

func (m *MockLiveLLM) Name() string {
	return "mock-live-llm"
}

func (m *MockLiveLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		// Not used in live mode
	}
}

func (m *MockLiveLLM) Connect(ctx context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
	if m.ConnectFunc != nil {
		return m.ConnectFunc(ctx, req)
	}
	return nil, fmt.Errorf("ConnectFunc not implemented")
}

func TestLiveAgentRun(t *testing.T) {
	ctx := context.Background()
	service := session.InMemoryService()
	queue := agent.NewLiveRequestQueue()

	// Channel to coordinate test steps
	receivedCh := make(chan string, 10)

	// Setup Mock Live Connection
	mockConn := &MockLiveConnection{
		SendFunc: func(req *model.LiveRequest) error {
			if req.Content != nil {
				t.Logf("Mock received content: %v", req.Content.Parts[0].Text)
			}
			return nil
		},
		ReceiveFunc: func() (*model.LLMResponse, error) {
			select {
			case msg := <-receivedCh:
				return &model.LLMResponse{
					Content: genai.NewContentFromText(msg, genai.RoleModel),
				}, nil
			case <-time.After(100 * time.Millisecond):
				// Keep alive / heartbeat or just block?
				// For test simple: block until closed or keep returning nil?
				// runLive loop handles nil? No, Receive returns error or blocks.
				// We can block here.
				time.Sleep(100 * time.Millisecond)
				return nil, nil // Return empty to skip yield? runLive handles nil content?
				// wrapper finalizes it.
			}
			return nil, nil
		},
		CloseFunc: func() error {
			return nil
		},
	}

	mockLLM := &MockLiveLLM{
		ConnectFunc: func(ctx context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
			return mockConn, nil
		},
	}

	agentConfig := llmagent.Config{
		Name:        "live_agent",
		Description: "Agent to test live streaming",
		Model:       mockLLM,
		Instruction: "You are a live agent.",
	}

	liveAgent, err := llmagent.New(agentConfig)
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        "test_app",
		Agent:          liveAgent,
		SessionService: service,
	})
	if err != nil {
		t.Fatalf("Failed to create runner: %v", err)
	}

	// Create session
	createResp, _ := service.Create(ctx, &session.CreateRequest{AppName: "test_app", UserID: "user"})
	sessionID := createResp.Session.ID()

	// Start Runner in background
	go func() {
		// Run with Bidi streaming
		runConfig := agent.RunConfig{
			StreamingMode:    agent.StreamingModeBidi,
			LiveRequestQueue: queue,
		}
		// Initial user message triggers the run
		userContent := genai.NewContentFromText("Hello", genai.RoleUser)

		for ev, err := range r.Run(ctx, "user", sessionID, userContent, runConfig) {
			if err != nil {
				t.Logf("Runner error: %v", err)
				return
			}
			if ev.LLMResponse.Content != nil {
				t.Logf("Runner received event: %v", ev.LLMResponse.Content.Parts[0].Text)
			}
		}
	}()

	// Simulate conversation
	// 1. Send content to model via queue
	err = queue.SendContent(genai.NewContentFromText("User input from queue", genai.RoleUser))
	if err != nil {
		t.Fatalf("Failed to send content: %v", err)
	}

	// 2. Simulate model sending response
	receivedCh <- "Hello from model"

	// Wait a bit
	time.Sleep(1 * time.Second)

	// Close queue
	queue.Close()
}
