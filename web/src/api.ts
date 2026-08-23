import type { MatchInput, MatchResponse } from "./types";
import { enableOfflineCache } from "./offline";

let worker: Worker | undefined;
let nextId = 1;
let fatalError: Error | undefined;
const pending = new Map<
  number,
  { resolve: (value: MatchResponse) => void; reject: (reason: Error) => void }
>();

function matcherWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./matcher.worker.ts", import.meta.url), { type: "module" });
  worker.addEventListener("message", (event) => {
    const message = event.data;
    if (message.type === "fatal") {
      const error = new Error(message.message);
      fatalError = error;
      pending.forEach(({ reject }) => reject(error));
      pending.clear();
      return;
    }
    if (message.type !== "result" && message.type !== "error") return;
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    if (message.type === "result") request.resolve(message.response as MatchResponse);
    else request.reject(new Error(message.message));
  });
  return worker;
}

export async function rankSessions(
  input: MatchInput,
  options: { topK?: number } = {},
): Promise<MatchResponse> {
  if (!input.abstract.trim()) throw new Error("An abstract is required.");
  if (fatalError) throw fatalError;
  const id = nextId++;
  const result = new Promise<MatchResponse>((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
  await enableOfflineCache();
  matcherWorker().postMessage({ type: "rank", id, input, topK: options.topK ?? 10 });
  return result;
}

export type { MatchInput, MatchResponse, RankedSession } from "./types";
