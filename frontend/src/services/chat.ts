export interface Citation {
  source_id?: string;
  source_name: string;
  section: string | null;
  source_url: string;
  document_version: string | null;
}

export interface ChatResult {
  response: string;
  conversation_id: string;
  refused: boolean;
  rate_limited: boolean;
  urgency?: "urgent" | "emergency" | null;
  issue_categories?: string[];
  citations?: Citation[];
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ConversationHistory {
  conversation_id: string;
  messages: ConversationMessage[];
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function ensureOk(response: Response): Promise<Response> {
  if (!response.ok) throw new Error("The server could not process the request.");
  return response;
}

export async function createConversation(): Promise<string> {
  const response = await ensureOk(await fetch(`${API_BASE}/api/conversations`, {
    method: "POST", headers: { "Content-Type": "application/json" }
  }));
  const result = await response.json() as { conversation_id: string };
  return result.conversation_id;
}

export async function getConversation(conversationId: string): Promise<ConversationHistory> {
  const response = await ensureOk(await fetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}`
  ));
  return response.json() as Promise<ConversationHistory>;
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await ensureOk(await fetch(
    `${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" }
  ));
}

export async function sendChat(
  message: string,
  sessionId: string,
  conversationId?: string | null,
  requestId: string = crypto.randomUUID(),
): Promise<ChatResult> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      conversation_id: conversationId || undefined,
      request_id: requestId,
    })
  });
  await ensureOk(response);
  return response.json() as Promise<ChatResult>;
}
