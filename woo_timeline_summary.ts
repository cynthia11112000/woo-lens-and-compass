export type ChatPlatform = "WhatsApp" | "Teams" | "SMS" | string;

export interface ChatMessage {
  sender?: string | null;
  timestamp?: string | null;
  body?: string | null;
  content?: string | null;
  isRedacted?: boolean;
  isSystemMessage?: boolean;
}

export interface ChatData {
  platform?: ChatPlatform;
  threadId?: string;
  chatName?: string;
  deelnemers?: string[];
  berichten?: ChatMessage[];
  messages?: ChatMessage[];
  sourceDocId?: string;
}

export interface DominantSpeaker {
  name: string;
  messages: number;
  share: number;
}

export type KeyEventType =
  | "crisis"
  | "deadline"
  | "course-shift"
  | "volume-spike"
  | "speaker-shift"
  | "redaction-spike";

export interface KeyEvent {
  type: KeyEventType;
  title: string;
  description: string;
  evidence: string[];
}

export interface RedactionHotspot {
  window: string;
  redactedMessages: number;
  totalMessages: number;
  note: string;
}

export interface RedactionAnalysis {
  redactedMessages: number;
  totalMessages: number;
  redactionRate: number;
  note: string;
  hotspots: RedactionHotspot[];
}

export interface WooTimelineSummaryEntry {
  date: string;
  totalMessages: number;
  totalConversations: number;
  dominantSpeakers: DominantSpeaker[];
  keyEvents: KeyEvent[];
  redactionAnalysis: RedactionAnalysis;
  narrative: string;
}

interface NormalizedMessage {
  dateKey: string;
  hour: number | null;
  sender: string;
  body: string;
  isRedacted: boolean;
  threadId: string;
}

const REDACTION_RE = /\[(?:GELAKT|REDACTED)(?::[^\]]*)?\]|\b5\.[12]\.\d[a-z]{0,2}\b|X{5,}/i;
const CRISIS_RE = /\b(crisis|spoed|urgent|escalat(?:ie|ed)|incident|paniek|nood|dringend)\b/i;
const DEADLINE_RE = /\b(deadline|uiterlijk|voor\s+\d{1,2}[:.]\d{2}|voor\s+maandag|voor\s+dinsdag|voor\s+woensdag|voor\s+donderdag|voor\s+vrijdag|eod|e\.o\.d\.)\b/i;
const COURSE_SHIFT_RE = /\b(koerswijziging|we\s+gaan\s+toch|besloten\s+om|nieuw\s+plan|draai|pivot|switch|in\s+plaats\s+van|herzien)\b/i;

function parseDateKey(raw: string): string | null {
  const text = raw.trim();
  const isoDate = text.match(/^(\d{4}-\d{2}-\d{2})/);
  if (isoDate) return isoDate[1];

  const slashDate = text.match(/^(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})$/);
  if (slashDate) {
    const day = slashDate[1].padStart(2, "0");
    const month = slashDate[2].padStart(2, "0");
    const year = slashDate[3].length === 2 ? `20${slashDate[3]}` : slashDate[3];
    return `${year}-${month}-${day}`;
  }

  return null;
}

function parseHour(raw: string): number | null {
  const hit = raw.match(/(?:T|\s|^)(\d{1,2}):(\d{2})(?::\d{2})?$/);
  if (!hit) return null;
  const hour = Number(hit[1]);
  if (Number.isNaN(hour) || hour < 0 || hour > 23) return null;
  return hour;
}

function asMessageList(chat: ChatData): ChatMessage[] {
  if (Array.isArray(chat.berichten)) return chat.berichten;
  if (Array.isArray(chat.messages)) return chat.messages;
  return [];
}

function normalizeMessages(chats: ChatData[]): NormalizedMessage[] {
  const normalized: NormalizedMessage[] = [];

  for (const chat of chats) {
    const messages = asMessageList(chat);
    const threadId = chat.threadId || chat.chatName || chat.sourceDocId || "unknown-thread";

    let threadDateHint: string | null = null;
    for (const msg of messages) {
      const ts = String(msg.timestamp || "").trim();
      if (!ts) continue;
      const maybeDate = parseDateKey(ts);
      if (maybeDate) {
        threadDateHint = maybeDate;
        break;
      }
    }

    for (const msg of messages) {
      const body = String(msg.body ?? msg.content ?? "").trim();
      if (!body) continue;

      const timestamp = String(msg.timestamp || "").trim();
      const directDate = parseDateKey(timestamp);
      const dateKey = directDate || (timestamp && threadDateHint ? threadDateHint : "onbekend");
      const hour = parseHour(timestamp);

      const sender = String(msg.sender || "Onbekend").trim() || "Onbekend";
      const isRedacted = Boolean(msg.isRedacted) || REDACTION_RE.test(body);

      normalized.push({
        dateKey,
        hour,
        sender,
        body,
        isRedacted,
        threadId,
      });
    }
  }

  return normalized;
}

function computeDominantSpeakers(messages: NormalizedMessage[]): DominantSpeaker[] {
  const counts = new Map<string, number>();
  for (const m of messages) {
    counts.set(m.sender, (counts.get(m.sender) || 0) + 1);
  }

  const total = messages.length || 1;
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([name, count]) => ({
      name,
      messages: count,
      share: Number((count / total).toFixed(3)),
    }));
}

function computeRedactionAnalysis(messages: NormalizedMessage[]): RedactionAnalysis {
  const redacted = messages.filter((m) => m.isRedacted);
  const total = messages.length;
  const rate = total === 0 ? 0 : redacted.length / total;

  const perHour = new Map<number, { redacted: number; total: number }>();
  for (const m of messages) {
    if (m.hour === null) continue;
    const bucket = perHour.get(m.hour) || { redacted: 0, total: 0 };
    bucket.total += 1;
    if (m.isRedacted) bucket.redacted += 1;
    perHour.set(m.hour, bucket);
  }

  const hotspots: RedactionHotspot[] = [...perHour.entries()]
    .filter(([, b]) => b.redacted >= 2 || (b.total >= 3 && b.redacted / b.total >= 0.6))
    .sort((a, b) => b[1].redacted - a[1].redacted)
    .slice(0, 3)
    .map(([hour, b]) => ({
      window: `${String(hour).padStart(2, "0")}:00-${String((hour + 1) % 24).padStart(2, "0")}:00`,
      redactedMessages: b.redacted,
      totalMessages: b.total,
      note:
        b.redacted / Math.max(1, b.total) >= 0.6
          ? "Veel onleesbare berichten in dit tijdsvenster."
          : "Opvallende redactiepiek in dit tijdsvenster.",
    }));

  let note = "Beperkte redactie in deze periode.";
  if (rate >= 0.5) note = "Zeer hoge redactie-intensiteit; inhoud is sterk afgeschermd.";
  else if (rate >= 0.3) note = "Hoge redactie-intensiteit met merkbare informatiegaten.";
  else if (rate >= 0.15) note = "Gemiddelde redactie-intensiteit; sommige context ontbreekt.";

  return {
    redactedMessages: redacted.length,
    totalMessages: total,
    redactionRate: Number(rate.toFixed(3)),
    note,
    hotspots,
  };
}

function detectKeywordEvent(
  type: KeyEventType,
  title: string,
  messages: NormalizedMessage[],
  re: RegExp,
  threshold: number,
): KeyEvent | null {
  const hits = messages.filter((m) => re.test(m.body));
  if (hits.length < threshold) return null;

  return {
    type,
    title,
    description: `${hits.length} berichten wijzen op ${title.toLowerCase()}.`,
    evidence: hits.slice(0, 3).map((m) => `${m.sender}: ${m.body.slice(0, 120)}`),
  };
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, n) => sum + n, 0) / values.length;
}

function makeNarrative(
  date: string,
  totalMessages: number,
  dominant: DominantSpeaker[],
  keyEvents: KeyEvent[],
  redaction: RedactionAnalysis,
): string {
  const lead = dominant[0]?.name || "Onbekend";
  const events = keyEvents.length
    ? keyEvents.map((e) => e.title.toLowerCase()).join(", ")
    : "geen uitgesproken kantelpunt";

  const hotspotText = redaction.hotspots[0]
    ? `Veel onleesbare berichten rond ${redaction.hotspots[0].window}.`
    : "Geen duidelijk redactie-hotspot per uur.";

  return `${date}: ${totalMessages} berichten, dominante spreker: ${lead}. Belangrijkste signaal: ${events}. ${hotspotText}`;
}

export function generateWooTimelineSummary(chats: ChatData[]): WooTimelineSummaryEntry[] {
  const allMessages = normalizeMessages(chats);
  const groups = new Map<string, NormalizedMessage[]>();

  for (const msg of allMessages) {
    const list = groups.get(msg.dateKey) || [];
    list.push(msg);
    groups.set(msg.dateKey, list);
  }

  const orderedDates = [...groups.keys()].sort((a, b) => {
    if (a === "onbekend") return 1;
    if (b === "onbekend") return -1;
    return a.localeCompare(b);
  });

  const summaries: WooTimelineSummaryEntry[] = [];

  for (let i = 0; i < orderedDates.length; i += 1) {
    const date = orderedDates[i];
    const dayMessages = groups.get(date) || [];
    const dominant = computeDominantSpeakers(dayMessages);
    const redaction = computeRedactionAnalysis(dayMessages);

    const keyEvents: KeyEvent[] = [];

    const crisis = detectKeywordEvent("crisis", "Crisis-moment", dayMessages, CRISIS_RE, 2);
    if (crisis) keyEvents.push(crisis);

    const deadline = detectKeywordEvent("deadline", "Deadline-druk", dayMessages, DEADLINE_RE, 2);
    if (deadline) keyEvents.push(deadline);

    const shift = detectKeywordEvent("course-shift", "Koerswijziging", dayMessages, COURSE_SHIFT_RE, 1);
    if (shift) keyEvents.push(shift);

    if (i > 0) {
      const previousTotals = summaries.slice(Math.max(0, i - 3), i).map((s) => s.totalMessages);
      const baseline = average(previousTotals);
      if (baseline >= 5 && dayMessages.length >= baseline * 1.6) {
        keyEvents.push({
          type: "volume-spike",
          title: "Activiteitspiek",
          description: `Berichtvolume steeg van gemiddeld ${baseline.toFixed(1)} naar ${dayMessages.length}.`,
          evidence: ["Plotselinge toename in berichtfrequentie."],
        });
      }

      const prevLead = summaries[i - 1]?.dominantSpeakers[0];
      const nowLead = dominant[0];
      if (prevLead && nowLead && prevLead.name !== nowLead.name && nowLead.share >= 0.35) {
        keyEvents.push({
          type: "speaker-shift",
          title: "Verschuiving in regie",
          description: `Dominante spreker verschoof van ${prevLead.name} naar ${nowLead.name}.`,
          evidence: ["Nieuwe stem bepaalt relatief groot deel van de conversatie."],
        });
      }
    }

    if (redaction.hotspots.length > 0) {
      keyEvents.push({
        type: "redaction-spike",
        title: "Redactiepiek",
        description: `Opvallend veel gelakte inhoud, met piek rond ${redaction.hotspots[0].window}.`,
        evidence: redaction.hotspots.map((h) => `${h.window}: ${h.redactedMessages}/${h.totalMessages} gelakt`),
      });
    }

    const conversations = new Set(dayMessages.map((m) => m.threadId)).size;

    summaries.push({
      date,
      totalMessages: dayMessages.length,
      totalConversations: conversations,
      dominantSpeakers: dominant,
      keyEvents,
      redactionAnalysis: redaction,
      narrative: makeNarrative(date, dayMessages.length, dominant, keyEvents, redaction),
    });
  }

  return summaries;
}
