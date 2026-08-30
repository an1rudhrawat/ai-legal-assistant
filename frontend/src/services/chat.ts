export interface ChatResult { response: string; refused: boolean; rate_limited: boolean; }
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function sendChat(message: string, sessionId: string): Promise<ChatResult> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId })
  });
  if (!response.ok) throw new Error("The server could not process the request.");
  return response.json() as Promise<ChatResult>;
}
