import { FormEvent, useEffect, useRef, useState } from "react";
import { useSpeechRecognition } from "./hooks/useSpeechRecognition";
import { speak, stopSpeaking } from "./hooks/useSpeechSynthesis";
import {
  ChatResult,
  ConversationMessage,
  createConversation,
  deleteConversation,
  getConversation,
  sendChat,
} from "./services/chat";
import { FormattedResponse, speechText } from "./utils/responseFormatting";

const DISCLAIMER = "This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.";
const SUGGESTIONS = [
  "The police will not register my FIR",
  "I received a court summons",
  "Explain a section of Indian law",
];

const sessionId = (() => {
  const saved = sessionStorage.getItem("legal-assistant-session");
  if (saved) return saved;
  const created = crypto.randomUUID();
  sessionStorage.setItem("legal-assistant-session", created);
  return created;
})();

function CitationList({ citations }: { citations: ChatResult["citations"] }) {
  if (!citations?.length) return null;
  return <section className="sources" aria-label="Sources">
    <h3>Sources</h3>
    <ul>
      {citations.map((citation) => {
        const url = /^(https?:\/\/)/i.test(citation.source_url) ? citation.source_url : null;
        return <li key={citation.source_id ?? `${citation.source_name}-${citation.section}`}>
          <span className="source-name">
            {url ? <a href={url} target="_blank" rel="noreferrer">{citation.source_name}</a> : citation.source_name}
          </span>
          {citation.section && <span className="source-detail">Section {citation.section}</span>}
          {citation.document_version && <span className="source-detail">Version {citation.document_version}</span>}
        </li>;
      })}
    </ul>
  </section>;
}

export default function App() {
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState<ChatResult | null>(null);
  const [answerMessageId, setAnswerMessageId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(
    () => sessionStorage.getItem("legal-assistant-conversation")
  );
  const [conversationMessages, setConversationMessages] = useState<ConversationMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pendingRequest = useRef<{ message: string; id: string } | null>(null);
  const loadedConversationId = useRef<string | null>(null);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const voice = useSpeechRecognition((transcript) => setMessage(transcript));

  useEffect(() => {
    if (!conversationId || loadedConversationId.current === conversationId) return;
    let active = true;
    loadedConversationId.current = conversationId;
    getConversation(conversationId)
      .then((history) => {
        if (active && Array.isArray(history.messages)) setConversationMessages(history.messages);
      })
      .catch((err) => {
        if (active) {
          loadedConversationId.current = null;
          setError(err instanceof Error ? err.message : "Unable to load this conversation.");
        }
      });
    return () => { active = false; };
  }, [conversationId]);

  useEffect(() => {
    const field = composerRef.current;
    if (!field) return;
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, 180)}px`;
  }, [message]);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [conversationMessages.length, loading]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const submittedMessage = message.trim();
    if (!submittedMessage || loading) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setAnswerMessageId(null);
    if (pendingRequest.current?.message !== submittedMessage) {
      pendingRequest.current = { message: submittedMessage, id: crypto.randomUUID() };
    }
    const createdAt = new Date().toISOString();
    setConversationMessages((previous) => [
      ...previous,
      { id: crypto.randomUUID(), role: "user", content: submittedMessage, created_at: createdAt },
    ]);
    setMessage("");

    try {
      const result = await sendChat(
        submittedMessage,
        sessionId,
        conversationId,
        pendingRequest.current.id,
      );
      pendingRequest.current = null;
      loadedConversationId.current = result.conversation_id;
      setConversationId(result.conversation_id);
      sessionStorage.setItem("legal-assistant-conversation", result.conversation_id);
      const assistantMessageId = crypto.randomUUID();
      setAnswer(result);
      setAnswerMessageId(assistantMessageId);
      setConversationMessages((previous) => [
        ...previous,
        { id: assistantMessageId, role: "assistant", content: result.response, created_at: new Date().toISOString() },
      ]);
      speak(speechText(result.response));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
      composerRef.current?.focus();
    }
  }

  async function startNewConversation() {
    try {
      const id = await createConversation();
      loadedConversationId.current = id;
      pendingRequest.current = null;
      setConversationId(id);
      sessionStorage.setItem("legal-assistant-conversation", id);
      setConversationMessages([]);
      setAnswer(null);
      setAnswerMessageId(null);
      setMessage("");
      setError(null);
      stopSpeaking();
      composerRef.current?.focus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start a new conversation.");
    }
  }

  async function clearConversation() {
    if (!conversationId || !window.confirm("Delete this conversation and its saved messages?")) return;
    try {
      await deleteConversation(conversationId);
      loadedConversationId.current = null;
      pendingRequest.current = null;
      sessionStorage.removeItem("legal-assistant-conversation");
      setConversationId(null);
      setConversationMessages([]);
      setAnswer(null);
      setAnswerMessageId(null);
      setMessage("");
      setError(null);
      stopSpeaking();
      composerRef.current?.focus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to delete this conversation.");
    }
  }

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#main" aria-label="Aadhikar AI home">
        <span className="brand-mark" aria-hidden="true">A</span>
        <span className="brand-copy"><strong>Aadhikar AI</strong><small>Indian legal information</small></span>
      </a>
      <div className="topbar-actions">
        <span className="privacy-note"><span className="status-dot" /> Private conversation</span>
        <button type="button" className="toolbar-button" onClick={startNewConversation} disabled={loading}>
          <span aria-hidden="true">＋</span> New chat
        </button>
        {conversationId && <button type="button" className="icon-button" onClick={clearConversation} disabled={loading} aria-label="Delete conversation" title="Delete conversation">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 6h12M8 6V4h4v2m-6 0 .7 10h6.6L14 6M8.5 9v4m3-4v4" /></svg>
        </button>}
      </div>
    </header>

    <main id="main" className="chat-layout">
      <section className={`chat-scroll${conversationMessages.length ? " has-messages" : " is-welcome"}`} aria-label="Conversation">
        {!conversationMessages.length ? <div className="welcome-panel">
          <div className="welcome-emblem" aria-hidden="true"><span>A</span></div>
          <p className="eyebrow">A clearer first step</p>
          <h1>How can I help?</h1>
          <p className="welcome-copy">Ask about an Indian legal issue, a document you received, or a section of law. I’ll use the available sources and tell you when they don’t cover your question.</p>
          <div className="suggestions" aria-label="Example questions">
            {SUGGESTIONS.map((suggestion) => <button key={suggestion} type="button" className="suggestion" onClick={() => { setMessage(suggestion); composerRef.current?.focus(); }}>
              <span>{suggestion}</span><span aria-hidden="true">↗</span>
            </button>)}
          </div>
          <p className="welcome-disclaimer">General legal information only. Not a substitute for advice from a licensed advocate.</p>
        </div> : <div className="message-list" aria-live="polite">
          {conversationMessages.map((item) => {
            const isUser = item.role === "user";
            const currentCitations = item.id === answerMessageId ? answer?.citations : undefined;
            return <article className={`message ${isUser ? "user-message" : "assistant-message"}`} key={item.id}>
              {!isUser && <span className="assistant-avatar" aria-hidden="true">A</span>}
              <div className="message-column">
                <div className="message-role">{isUser ? "You" : "Aadhikar AI"}</div>
                {item.id === answerMessageId && answer?.urgency && <p className="urgency" role="alert">
                  {answer.urgency === "emergency" ? "Immediate safety concern" : "Time-sensitive legal situation"}
                </p>}
                {isUser ? <p className="user-text">{item.content}</p> : <FormattedResponse text={item.content} />}
                {currentCitations && <CitationList citations={currentCitations} />}
              </div>
            </article>;
          })}
          {loading && <div className="message assistant-message typing-row" role="status" aria-label="Aadhikar AI is preparing a response">
            <span className="assistant-avatar" aria-hidden="true">A</span>
            <div className="typing-indicator"><span /><span /><span /></div>
          </div>}
          <div ref={conversationEndRef} className="conversation-end" />
        </div>}
        {!conversationMessages.length && <div ref={conversationEndRef} className="conversation-end" />}
      </section>

      <div className="composer-wrap">
        {error && <p className="error-message composer-error" role="alert">{error}</p>}
        {!voice.supported && <p className="voice-hint">Voice input isn’t available in this browser. You can still type your question.</p>}
        {voice.error && <p className="error-message voice-error" role="alert">{voice.error}</p>}
        <form className="composer" onSubmit={submit} aria-label="Ask a legal question">
          <label className="visually-hidden" htmlFor="question">Your question</label>
          <textarea
            ref={composerRef}
            id="question"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Message Aadhikar AI…"
            rows={1}
            maxLength={3000}
            disabled={loading}
          />
          <div className="composer-footer">
            <div className="composer-tools">
              <button type="button" className={`tool-button${voice.isListening ? " listening" : ""}`} onClick={voice.start} disabled={!voice.supported || voice.isListening || loading} aria-label={voice.isListening ? "Listening" : "Use microphone"} title={voice.isListening ? "Listening" : "Use microphone"}>
                <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="7" y="3" width="6" height="10" rx="3" /><path d="M5 9.5a5 5 0 0 0 10 0M10 15v3m-3 0h6" /></svg>
              </button>
              <button type="button" className="tool-button" onClick={stopSpeaking} aria-label="Stop speaking" title="Stop speaking">
                <svg viewBox="0 0 20 20" aria-hidden="true"><rect x="5" y="5" width="10" height="10" rx="2" /></svg>
              </button>
              <span className="keyboard-hint">Enter to send · Shift+Enter for a new line</span>
            </div>
            <div className="composer-submit">
              <span className="character-count">{message.length}/3000</span>
              <button type="submit" className="send-button" disabled={loading || !message.trim()} aria-label={loading ? "Preparing response" : "Send message"}>
                {loading ? <span className="send-spinner" aria-hidden="true" /> : <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 15V4m-4 4 4-4 4 4" /></svg>}
              </button>
            </div>
          </div>
        </form>
        <p className="composer-disclaimer">{DISCLAIMER}</p>
      </div>
    </main>
  </div>;
}
