import "./style.css";
import { enableOfflineCache } from "./offline";
import type { MatchResponse } from "./types";

void enableOfflineCache();

const form = document.querySelector<HTMLFormElement>("#matcher-form")!;
const title = document.querySelector<HTMLInputElement>("#title")!;
const abstract = document.querySelector<HTMLTextAreaElement>("#abstract")!;
const characterCount = document.querySelector<HTMLElement>("#character-count")!;
const submit = document.querySelector<HTMLButtonElement>("#submit")!;
const status = document.querySelector<HTMLElement>("#status")!;
const section = document.querySelector<HTMLElement>("#results-section")!;
const results = document.querySelector<HTMLOListElement>("#results")!;
const modelInfo = document.querySelector<HTMLElement>("#model-info")!;
const ambiguity = document.querySelector<HTMLElement>("#ambiguity")!;

const worker = new Worker(new URL("./matcher.worker.ts", import.meta.url), { type: "module" });
let ready = false;
let requestId = 0;

function confidenceText(value: "high" | "medium" | "low"): string {
  if (value === "high") return "focused shortlist";
  if (value === "low") return "broad shortlist";
  return "moderate ambiguity";
}

function render(response: MatchResponse): void {
  results.replaceChildren();
  modelInfo.textContent = `EGU${String(response.catalogueYear).slice(-2)} · ${response.results.length} shown · local inference`;
  const uncertain = response.results.length > 1 && response.results[0].similarity - response.results[1].similarity < 0.025;
  const dynamic = response.selection.mode === "dynamic";
  const setSize = response.selection.predictionSetSize ?? response.results.length;
  const selectionMessage = dynamic
    ? `This abstract produced a calibrated set of ${setSize} plausible session${setSize === 1 ? "" : "s"}, targeting ${Math.round((response.selection.targetCoverage ?? 0) * 100)}% historical assignment coverage.${response.selection.truncated ? ` The first ${response.selection.shownCount} are displayed.` : ""}`
    : `Showing the first ${response.results.length} ranked sessions.`;
  ambiguity.hidden = false;
  ambiguity.textContent = `${selectionMessage}${uncertain ? " Several leading sessions have nearly equal scores, so their descriptions deserve particular attention." : ""}`;

  response.results.forEach((match) => {
    const item = document.createElement("li");
    item.className = "result-card";
    item.style.setProperty("--division", match.color ?? "#52656f");

    const rank = document.createElement("span");
    rank.className = "rank-number";
    rank.textContent = String(match.rank).padStart(2, "0");

    const content = document.createElement("div");
    const heading = document.createElement("h3");
    const link = document.createElement("a");
    link.href = match.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `${match.code} · ${match.title}`;
    heading.append(link);
    const meta = document.createElement("p");
    meta.className = "result-meta";
    const division = document.createElement("span");
    division.className = "division-pill";
    division.textContent = match.divisions.join(" · ") || "Cross-programme session";
    meta.append(division);
    content.append(heading, meta);

    const score = document.createElement("div");
    score.className = "score-block";
    const scoreValue = document.createElement("strong");
    scoreValue.textContent = match.similarity.toFixed(3);
    const scoreLabel = document.createElement("span");
    scoreLabel.textContent = "relative score";
    const confidence = document.createElement("small");
    confidence.className = "confidence-badge";
    confidence.textContent = confidenceText(match.confidence);
    score.append(scoreValue, scoreLabel, confidence);

    item.append(rank, content, score);
    if (match.description) {
      const details = document.createElement("details");
      details.className = "description-details";
      const summary = document.createElement("summary");
      summary.textContent = "Read official session description";
      const description = document.createElement("p");
      description.textContent = match.description;
      details.append(summary, description);
      item.append(details);
    }
    results.append(item);
  });
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

abstract.addEventListener("input", () => {
  characterCount.textContent = `${abstract.value.length.toLocaleString()} / 30,000`;
});

worker.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "status") status.textContent = message.message;
  if (message.type === "ready") {
    ready = true;
    submit.disabled = false;
    document.querySelectorAll<HTMLElement>("[data-catalogue-year]").forEach((element) => {
      element.textContent = String(message.year).slice(-2);
    });
    document.querySelectorAll<HTMLElement>("[data-session-count]").forEach((element) => {
      element.textContent = Number(message.sessionCount).toLocaleString();
    });
    status.textContent = `Ready · EGU${String(message.year).slice(-2)} · ${String(message.backend).toUpperCase()} · text remains on this device`;
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
