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

package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for this example
	},
}

type ClientMessage struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

func main() {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY environment variable is not set")
	}

	r := mux.NewRouter()

	// WebSocket handler with userId and sessionId parameters
	r.HandleFunc("/ws/{userId}/{sessionId}", wsHandler)

	// Static file server
	staticDir := "static"
	r.PathPrefix("/").Handler(http.FileServer(http.Dir(staticDir)))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("Server starting on http://localhost:%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, r))
}

func wsHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userId"]
	sessionID := vars["sessionId"]

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Failed to upgrade connection: %v", err)
		return
	}
	defer conn.Close()

	ctx := context.Background()
	apiKey := os.Getenv("GOOGLE_API_KEY")

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-native-audio-preview-12-2025", &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Printf("Failed to create model: %v", err)
		return
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "live_agent",
		Model:       model,
		Description: "A live agent that echoes what you say.",
		Instruction: "You are a live assistant. Respond briefly to the user.",
	})
	if err != nil {
		log.Printf("Failed to create agent: %v", err)
		return
	}

	sessionService := session.InMemoryService()
	runn, err := runner.New(runner.Config{
		AppName:        "live_sample",
		Agent:          a,
		SessionService: sessionService,
	})
	if err != nil {
		log.Printf("Failed to create runner: %v", err)
		return
	}

	// Create session for the user
	_, err = sessionService.Create(ctx, &session.CreateRequest{
		AppName:   "live_sample",
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		log.Printf("Session creation (might already exist): %v", err)
	}

	liveRequestQueue := agent.NewLiveRequestQueue()

	runConfig := agent.RunConfig{
		StreamingMode: agent.StreamingModeBidi,
		LiveConnectConfig: &genai.LiveConnectConfig{
			ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityAudio},
			SpeechConfig: &genai.SpeechConfig{
				VoiceConfig: &genai.VoiceConfig{
					PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
						VoiceName: "Aoide",
					},
				},
			},
		},
	}

	// Channel to signal the reading loop to stop
	done := make(chan struct{})

	// Start receiving ADK events and sending them to the WebSocket
	go func() {
		defer close(done)
		userContent := genai.NewContentFromText("Start session", genai.RoleUser)

		for ev, err := range runn.Run(ctx, userID, sessionID, userContent, runConfig) {
			if err != nil {
				log.Printf("Runner error: %v", err)
				return
			}

			// Convert ADK event to JSON and send to client
			evJSON, err := json.Marshal(ev)
			if err != nil {
				log.Printf("Failed to marshal event: %v", err)
				continue
			}

			if err := conn.WriteMessage(websocket.TextMessage, evJSON); err != nil {
				log.Printf("WebSocket write error: %v", err)
				return
			}
		}
	}()

	// Read from WebSocket and send to the ADK LiveRequestQueue
	for {
		messageType, p, err := conn.ReadMessage()
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		if messageType == websocket.BinaryMessage {
			// Binary data is assumed to be PCM audio from the client
			err := liveRequestQueue.SendContent(&genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "audio/pcm;rate=16000",
							Data:     p,
						},
					},
				},
			})
			if err != nil {
				log.Printf("Error sending audio to queue: %v", err)
			}
		} else if messageType == websocket.TextMessage {
			var msg ClientMessage
			if err := json.Unmarshal(p, &msg); err != nil {
				log.Printf("Failed to unmarshal text message: %v", err)
				continue
			}

			switch msg.Type {
			case "text":
				err := liveRequestQueue.SendContent(genai.NewContentFromText(msg.Text, genai.RoleUser))
				if err != nil {
					log.Printf("Error sending text to queue: %v", err)
				}
			case "image":
				data, err := base64.StdEncoding.DecodeString(msg.Data)
				if err != nil {
					log.Printf("Failed to decode base64 image: %v", err)
					continue
				}
				err = liveRequestQueue.SendContent(&genai.Content{
					Role: "user",
					Parts: []*genai.Part{
						{
							InlineData: &genai.Blob{
								MIMEType: msg.MimeType,
								Data:     data,
							},
						},
					},
				})
				if err != nil {
					log.Printf("Error sending image to queue: %v", err)
				}
			}
		}

		select {
		case <-done:
			return
		default:
		}
	}

	liveRequestQueue.Close()
}
