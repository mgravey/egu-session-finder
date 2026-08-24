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
const moreResultsWrap = document.querySelector<HTMLElement>("#more-results-wrap")!;
const moreResults = document.querySelector<HTMLButtonElement>("#more-results")!;
const moreResultsLabel = moreResults.querySelector<HTMLElement>("span")!;
const moreResultsNote = document.querySelector<HTMLElement>("#more-results-note")!;

const worker = new Worker(new URL("./matcher.worker.ts", import.meta.url), { type: "module" });
let ready = false;
let requestId = 0;
let lastResponse: MatchResponse | null = null;
let lastInput: { title: string; abstract: string } | null = null;
let expansionRequest = false;

function confidenceText(value: "high" | "medium" | "low"): string {
  if (value === "high") return "focused shortlist";
  if (value === "low") return "broad shortlist";
  return "moderate ambiguity";
}

function transparentDivisionColor(color: string, alpha: number): string {
  const match = /^#([0-9a-f]{6})$/i.exec(color);
  if (!match) return `color-mix(in srgb, ${color} ${Math.round(alpha * 100)}%, transparent)`;
  const value = Number.parseInt(match[1], 16);
  return `rgba(${value >> 16}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

function cardDivisionStyle(item: HTMLElement, colors: string[]): void {
  const safeColors = colors.length ? colors : ["#52656f"];
  const primary = safeColors[0];
  item.style.setProperty("--division", primary);
  if (safeColors.length <= 3) {
    const stops = safeColors.map((color, index) => {
      const position = safeColors.length === 1 ? 0 : (index / (safeColors.length - 1)) * 100;
      return `${transparentDivisionColor(color, index === 0 ? 0.18 : 0.13)} ${position}%`;
    });
    stops.push(`${transparentDivisionColor(safeColors.at(-1)!, 0.045)} 100%`);
    item.style.setProperty("--session-background", `linear-gradient(110deg, ${stops.join(", ")})`);
  } else {
    item.classList.add("many-divisions");
    item.style.setProperty(
      "--session-background",
      `linear-gradient(110deg, ${transparentDivisionColor(primary, 0.18)}, ${transparentDivisionColor(primary, 0.045)})`,
    );
  }
  if (safeColors.length > 1) {
    item.classList.add("multi-division");
    const width = 100 / safeColors.length;
    const segments = safeColors.flatMap((color, index) => [
      `${color} ${(index * width).toFixed(3)}%`,
      `${color} ${((index + 1) * width).toFixed(3)}%`,
    ]);
    const direction = safeColors.length > 3 ? "180deg" : "90deg";
    item.style.setProperty(
      "--division-spectrum",
      `linear-gradient(${direction}, ${segments.join(", ")})`,
    );
  }
}

function updateMoreControl(response: MatchResponse): void {
  const total = response.selection.candidateCount;
  const calibratedTotal = response.selection.predictionSetSize;
  const shown = response.selection.shownCount;
  const remaining = Math.max(0, total - shown);
  moreResultsWrap.hidden = remaining === 0;
  if (remaining === 0) return;
  const increment = Math.min(10, remaining);
  moreResultsLabel.textContent = `Show next ${increment}`;
  if (calibratedTotal !== undefined && shown < calibratedTotal) {
    moreResultsNote.textContent = `${shown} of ${calibratedTotal} calibrated candidates shown`;
  } else if (calibratedTotal !== undefined && shown === calibratedTotal) {
    moreResultsNote.textContent = `Complete calibrated shortlist shown · lower-ranked sessions remain`;
  } else if (calibratedTotal !== undefined) {
    moreResultsNote.textContent = `${shown} of ${total} sessions shown · now beyond the calibrated shortlist`;
  } else {
    moreResultsNote.textContent = `${shown} of ${total} sessions shown`;
  }
  moreResults.disabled = false;
}

function render(response: MatchResponse, shouldScroll = true): void {
  lastResponse = response;
  results.replaceChildren();
  modelInfo.textContent = `EGU${String(response.catalogueYear).slice(-2)} · ${response.results.length} shown · local inference`;
  const uncertain = response.results.length > 1 && response.results[0].similarity - response.results[1].similarity < 0.025;
  const dynamic = response.selection.predictionSetSize !== undefined;
  const setSize = response.selection.predictionSetSize ?? response.results.length;
  const selectionMessage = dynamic
    ? `This abstract produced a calibrated set of ${setSize} plausible session${setSize === 1 ? "" : "s"}, targeting ${Math.round((response.selection.targetCoverage ?? 0) * 100)}% historical assignment coverage.${response.selection.truncated ? ` The first ${response.selection.shownCount} are displayed.` : ""}`
    : `Showing the first ${response.results.length} ranked sessions.`;
  ambiguity.hidden = false;
  ambiguity.textContent = `${selectionMessage}${uncertain ? " Several leading sessions have nearly equal scores, so their descriptions deserve particular attention." : ""}`;

  response.results.forEach((match) => {
    const item = document.createElement("li");
    item.className = "result-card";
    const divisionColors = match.divisionColors?.length
      ? match.divisionColors
      : [match.color ?? "#52656f"];
    const visualDivisionCodes = match.visualDivisionCodes?.length
      ? match.visualDivisionCodes
      : match.divisions.map((value) => /\(([A-Z][A-Z0-9-]*)\)\s*$/.exec(value)?.[1] ?? value);
    cardDivisionStyle(item, divisionColors);

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
    division.textContent = visualDivisionCodes.join(" · ") || "Cross-programme session";
    division.title = match.divisions.join(" · ");
    meta.append(division);
    if (match.coOrganizerLabel) {
      const coOrganizer = document.createElement("span");
      coOrganizer.className = "coorganizer-label";
      coOrganizer.textContent = `Co-organised by ${match.coOrganizerLabel}`;
      meta.append(coOrganizer);
    }
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
  updateMoreControl(response);
  section.hidden = false;
  if (shouldScroll) section.scrollIntoView({ behavior: "smooth", block: "start" });
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
    const expanding = expansionRequest;
    expansionRequest = false;
    render(message.response as MatchResponse, !expanding);
    submit.disabled = false;
    status.textContent = expanding
      ? "Additional candidates displayed from the same private match."
      : "Matching complete. Your text was processed only in this browser.";
  }
  if ((message.type === "error" || message.type === "fatal") && (!message.id || message.id === requestId)) {
    expansionRequest = false;
    if (lastResponse) updateMoreControl(lastResponse);
    status.textContent = message.message;
    submit.disabled = !ready;
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!ready || !abstract.value.trim()) return;
  requestId += 1;
  lastResponse = null;
  lastInput = { title: title.value, abstract: abstract.value };
  expansionRequest = false;
  moreResultsWrap.hidden = true;
  submit.disabled = true;
  status.textContent = "Embedding and comparing your abstract locally…";
  worker.postMessage({
    type: "rank",
    id: requestId,
    input: lastInput,
  });
});

moreResults.addEventListener("click", () => {
  if (!ready || !lastResponse || !lastInput) return;
  const total = lastResponse.selection.candidateCount;
  const topK = Math.min(lastResponse.selection.shownCount + 10, total);
  if (topK <= lastResponse.selection.shownCount) return;
  requestId += 1;
  expansionRequest = true;
  moreResults.disabled = true;
  submit.disabled = true;
  status.textContent = "Reusing the private match to reveal more candidates…";
  worker.postMessage({ type: "rank", id: requestId, input: lastInput, topK });
});
