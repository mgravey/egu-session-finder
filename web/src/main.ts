import "./style.css";
import { enableOfflineCache } from "./offline";
import type { MatchResponse } from "./types";

void enableOfflineCache();

const form = document.querySelector<HTMLFormElement>("#matcher-form")!;
const title = document.querySelector<HTMLInputElement>("#title")!;
const abstract = document.querySelector<HTMLTextAreaElement>("#abstract")!;
const submit = document.querySelector<HTMLButtonElement>("#submit")!;
const status = document.querySelector<HTMLElement>("#status")!;
const section = document.querySelector<HTMLElement>("#results-section")!;
const results = document.querySelector<HTMLOListElement>("#results")!;
const modelInfo = document.querySelector<HTMLElement>("#model-info")!;
const ambiguity = document.querySelector<HTMLElement>("#ambiguity")!;

const worker = new Worker(new URL("./matcher.worker.ts", import.meta.url), { type: "module" });
let ready = false;
let requestId = 0;

function render(response: MatchResponse): void {
  results.replaceChildren();
  modelInfo.textContent = `EGU ${response.catalogueYear} · ${response.modelVersion}`;
  const uncertain =
    response.results.length > 1 &&
    response.results[0].similarity - response.results[1].similarity < 0.025;
  const dynamic = response.selection.mode === "dynamic";
  const selectionMessage = dynamic
    ? `The score-calibrated set contains ${response.selection.predictionSetSize} session${response.selection.predictionSetSize === 1 ? "" : "s"}, targeting ${Math.round((response.selection.targetCoverage ?? 0) * 100)}% historical assignment coverage.${response.selection.truncated ? ` The first ${response.selection.shownCount} are displayed.` : ""}`
    : "";
  ambiguity.hidden = !uncertain && !dynamic;
  ambiguity.textContent = [
    selectionMessage,
    uncertain ? "Several sessions are similarly relevant. Review the descriptions rather than relying only on rank." : "",
  ].filter(Boolean).join(" ");
  response.results.forEach((match) => {
    const item = document.createElement("li");
    item.className = "result-card";
    item.style.setProperty("--division", match.color ?? "#315b70");
    const heading = document.createElement("h3");
    const link = document.createElement("a");
    link.href = match.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `${match.code} · ${match.title}`;
    heading.append(link);
    const meta = document.createElement("p");
    meta.className = "result-meta";
    meta.textContent = `${match.divisions.join(" · ") || "Cross-programme session"} · similarity ${match.similarity.toFixed(3)}`;
    item.append(heading, meta);
    if (match.description) {
      const description = document.createElement("p");
      description.textContent = match.description;
      item.append(description);
    }
    results.append(item);
  });
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

worker.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "status") status.textContent = message.message;
  if (message.type === "ready") {
    ready = true;
    submit.disabled = false;
    status.textContent = `Ready for EGU ${message.year} · ${message.backend.toUpperCase()} · text remains on this device`;
  }
  if (message.type === "result" && message.id === requestId) {
    render(message.response as MatchResponse);
    submit.disabled = false;
    status.textContent = "Matching complete. Your text was processed only in this browser.";
  }
  if ((message.type === "error" || message.type === "fatal") && (!message.id || message.id === requestId)) {
    status.textContent = message.message;
    submit.disabled = !ready;
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!ready || !abstract.value.trim()) return;
  requestId += 1;
  submit.disabled = true;
  status.textContent = "Embedding and comparing your abstract locally…";
  worker.postMessage({
    type: "rank",
    id: requestId,
    input: { title: title.value, abstract: abstract.value },
  });
});
