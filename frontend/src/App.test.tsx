import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

describe("App", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
    vi.stubGlobal("crypto", { randomUUID: () => "browser-test-session" });
    vi.stubGlobal("speechSynthesis", { cancel: vi.fn(), speak: vi.fn() });
  });

  it("renders the persistent disclaimer and typed-input fallback", () => {
    render(<App />);
    expect(screen.getByText(/general legal information, not legal advice/i)).toBeInTheDocument();
    expect(screen.getByText(/Typed questions remain available/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /use microphone/i })).toBeDisabled();
  });

  it("displays a successful response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({ response: "An FIR is a police record.", refused: false, rate_limited: false }) }));
    render(<App />);
    fireEvent.change(screen.getByLabelText(/your question/i), { target: { value: "What is an FIR?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(screen.getByText("An FIR is a police record.")).toBeInTheDocument());
  });

  it("enables the microphone control when SpeechRecognition is available", () => {
    class MockRecognition {
      lang = ""; continuous = false; interimResults = false;
      onresult = null; onerror = null; onend = null;
      start = vi.fn(); stop = vi.fn();
      addEventListener = vi.fn(); removeEventListener = vi.fn(); dispatchEvent = vi.fn();
    }
    window.SpeechRecognition = MockRecognition as unknown as SpeechRecognitionConstructor;
    render(<App />);
    expect(screen.getByRole("button", { name: /use microphone/i })).toBeEnabled();
  });

  it("displays a legal refusal returned by the backend", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({ response: "I can only help with Indian legal terminology.", refused: true, rate_limited: false }) }));
    render(<App />);
    fireEvent.change(screen.getByLabelText(/your question/i), { target: { value: "Tell me a joke" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(screen.getByText(/I can only help with Indian legal terminology/i)).toBeInTheDocument());
  });
});
