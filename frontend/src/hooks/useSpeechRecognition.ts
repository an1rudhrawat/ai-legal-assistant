import { useMemo, useState } from "react";

export function useSpeechRecognition(onFinalTranscript: (text: string) => void) {
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const supported = useMemo(() => Boolean(window.SpeechRecognition || window.webkitSpeechRecognition), []);

  function start() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) { setError("Speech recognition is not supported in this browser. You can type your question instead."); return; }
    const recognition = new Recognition();
    recognition.lang = "en-IN";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onresult = (event) => onFinalTranscript(event.results[0][0].transcript.trim());
    recognition.onerror = () => { setError("I could not understand the audio. Please try again or type your question."); setIsListening(false); };
    recognition.onend = () => setIsListening(false);
    setError(null); setIsListening(true); recognition.start();
  }
  return { supported, isListening, error, start };
}
