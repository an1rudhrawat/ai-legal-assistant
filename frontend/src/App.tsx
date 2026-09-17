import { FormEvent, useState } from "react";
import { useSpeechRecognition } from "./hooks/useSpeechRecognition";
import { speak } from "./hooks/useSpeechSynthesis";
import { ChatResult, sendChat } from "./services/chat";

const DISCLAIMER = "This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.";
const sessionId = (() => {
  const saved = sessionStorage.getItem("legal-assistant-session");
  if (saved) return saved;
  const created = crypto.randomUUID(); sessionStorage.setItem("legal-assistant-session", created); return created;
})();

export default function App() {
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState<ChatResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const voice = useSpeechRecognition((transcript) => setMessage(transcript));

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!message.trim() || loading) return;
    setLoading(true); setError(null); setAnswer(null);
    try {
      const result = await sendChat(message.trim(), sessionId);
      setAnswer(result); speak(result.response);
    } catch (err) { setError(err instanceof Error ? err.message : "Something went wrong. Please try again."); }
    finally { setLoading(false); }
  }

  return <main className="app-shell">
    <header><h1>Legal First Responder</h1><p>Understand Indian legal situations, practical next steps, and legal terms.</p></header>
    <aside className="disclaimer" role="note">{DISCLAIMER}</aside>
    <form onSubmit={submit} aria-label="Legal question form">
      <label htmlFor="question">Your question</label>
      <textarea id="question" value={message} onChange={(e) => setMessage(e.target.value)} placeholder="For example: The police will not register my FIR." rows={5} />
      <div className="actions">
        <button type="button" onClick={voice.start} disabled={!voice.supported || voice.isListening}>{voice.isListening ? "Listening…" : "Use microphone"}</button>
        <button type="submit" disabled={loading || !message.trim()}>{loading ? "Thinking…" : "Ask"}</button>
      </div>
      {!voice.supported && <p className="hint">Speech recognition is unavailable in this browser. Typed questions remain available.</p>}
      {voice.error && <p className="error" role="alert">{voice.error}</p>}
    </form>
    {error && <p className="error" role="alert">{error}</p>}
    {answer && <section className="answer" aria-live="polite">
      {answer.urgency && <p className="urgency" role="alert">{answer.urgency === "emergency" ? "Immediate safety concern" : "Time-sensitive legal situation"}</p>}
      <h2>Response</h2><p>{answer.response}</p>
      {!!answer.citations?.length && <section className="sources"><h3>Sources</h3><ul>{answer.citations.map((citation) => <li key={`${citation.source_name}-${citation.section}`}>
        {citation.source_url ? <a href={citation.source_url} target="_blank" rel="noreferrer">{citation.source_name}</a> : citation.source_name}
        {citation.section && ` — Section ${citation.section}`}{citation.document_version && ` (${citation.document_version})`}
      </li>)}</ul></section>}
    </section>}
  </main>;
}
