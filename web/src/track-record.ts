import record from "./track-record.json";

interface YearRecord {
  year: number;
  trainingYears: number[];
  calibrationYear: number;
  abstracts: number;
  candidateSessions: number;
  model: number[];
  shortlist: { coverage: number; medianSize: number };
}

const SVG_NS = "http://www.w3.org/2000/svg";
const years = record.years as YearRecord[];
const percent = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;
const label = (year: number) => `EGU${String(year).slice(-2)}`;

function trainedOn(item: YearRecord): string {
  const first = item.trainingYears[0];
  const last = item.trainingYears[item.trainingYears.length - 1];
  return first === last ? String(first) : `${first}–${String(last).slice(-2)}`;
}

function svg<K extends keyof SVGElementTagNameMap>(
  name: K,
  attributes: Record<string, string | number>,
  parent?: Element,
): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  parent?.append(element);
  return element;
}

function renderTable(body: HTMLElement): void {
  body.replaceChildren();
  for (const item of [...years].reverse()) {
    const row = document.createElement("tr");
    const cells = [
      label(item.year),
      trainedOn(item),
      ...[1, 2, 5, 10, 20].map((k) => percent(item.model[k - 1], 1)),
      percent(item.shortlist.coverage, 1),
      `${item.shortlist.medianSize} sessions`,
    ];
    cells.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.setAttribute("scope", "row");
      cell.textContent = value;
      row.append(cell);
    });
    body.append(row);
  }
}

function renderChart(container: HTMLElement, tooltip: HTMLElement, item: YearRecord): void {
  // Draw in CSS pixels so text keeps its size on every screen width.
  const width = Math.max(300, Math.round(container.clientWidth || 720));
  const compact = width < 560;
  const height = compact ? 250 : 320;
  const margin = { top: 16, right: compact ? 48 : 64, bottom: 40, left: 44 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const ranks = item.model.length;
  const x = (rank: number) => margin.left + ((rank - 1) / (ranks - 1)) * plotWidth;
  const y = (value: number) => margin.top + (1 - value) * plotHeight;

  const root = svg("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `${label(item.year)}: share of abstracts whose session is within the top k suggestions, from k = 1 to ${ranks}`,
  });
  const grid = svg("g", { class: "chart-grid" }, root);
  for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
    svg("line", { x1: margin.left, x2: margin.left + plotWidth, y1: y(tick), y2: y(tick) }, grid);
    const text = svg("text", { x: margin.left - 10, y: y(tick) + 4, "text-anchor": "end" }, grid);
    text.textContent = percent(tick);
  }
  for (const rank of (compact ? [1, 5, 10, 15, 20] : [1, 2, 5, 10, 15, 20]).filter((value) => value <= ranks)) {
    const text = svg("text", { x: x(rank), y: margin.top + plotHeight + 22, "text-anchor": "middle" }, grid);
    text.textContent = String(rank);
  }
  const axisTitle = svg("text", { x: margin.left + plotWidth / 2, y: height - 2, "text-anchor": "middle", class: "axis-title" }, grid);
  axisTitle.textContent = "Number of suggestions read (k)";

  const values = item.model;
  const path = values.map((value, index) => `${index ? "L" : "M"}${x(index + 1)},${y(value)}`).join("");
  svg("path", { d: `${path}L${x(ranks)},${y(0)}L${x(1)},${y(0)}Z`, class: "series-area" }, root);
  svg("path", { d: path, class: "series-line series-model" }, root);
  const last = values[ranks - 1];
  svg("circle", { cx: x(ranks), cy: y(last), r: 4.5, class: "series-dot series-model" }, root);
  const endLabel = svg("text", { x: x(ranks) + 10, y: y(last) + 4, class: "series-label" }, root);
  endLabel.textContent = percent(last);

  const crosshair = svg("line", { class: "crosshair", y1: margin.top, y2: margin.top + plotHeight, visibility: "hidden" }, root);
  const marker = svg("circle", { r: 4.5, class: "series-dot series-model", visibility: "hidden" }, root);
  const hit = svg("rect", {
    x: margin.left - 12,
    y: margin.top,
    width: plotWidth + 24,
    height: plotHeight,
    class: "hit-area",
    tabindex: 0,
  }, root);

  const show = (rank: number) => {
    const position = x(rank);
    crosshair.setAttribute("x1", String(position));
    crosshair.setAttribute("x2", String(position));
    crosshair.setAttribute("visibility", "visible");
    marker.setAttribute("cx", String(position));
    marker.setAttribute("cy", String(y(values[rank - 1])));
    marker.setAttribute("visibility", "visible");
    const heading = document.createElement("p");
    heading.textContent = `${label(item.year)} · top ${rank}`;
    const row = document.createElement("div");
    const value = document.createElement("strong");
    value.textContent = percent(values[rank - 1], 1);
    const name = document.createElement("span");
    name.textContent = rank === 1 ? "first suggestion is right" : `right session within the top ${rank}`;
    row.append(value, name);
    tooltip.replaceChildren(heading, row);
    tooltip.hidden = false;
    const bounds = container.getBoundingClientRect();
    const left = (position / width) * bounds.width;
    tooltip.style.left = `${Math.min(Math.max(left + 14, 8), bounds.width - 210)}px`;
    tooltip.style.top = `${container.offsetTop + 12}px`;
  };
  const hide = () => {
    crosshair.setAttribute("visibility", "hidden");
    marker.setAttribute("visibility", "hidden");
    tooltip.hidden = true;
  };
  let focusedRank = 10;
  hit.addEventListener("pointermove", (event) => {
    const bounds = hit.getBoundingClientRect();
    const fraction = (event.clientX - bounds.left - (12 / (plotWidth + 24)) * bounds.width) / ((plotWidth / (plotWidth + 24)) * bounds.width);
    focusedRank = Math.min(ranks, Math.max(1, Math.round(fraction * (ranks - 1)) + 1));
    show(focusedRank);
  });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => show(focusedRank));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    focusedRank = Math.min(ranks, Math.max(1, focusedRank + (event.key === "ArrowRight" ? 1 : -1)));
    show(focusedRank);
  });
  container.replaceChildren(root);
}

export function initTrackRecord(): void {
  const chart = document.querySelector<HTMLElement>("#rank-chart");
  const tooltip = document.querySelector<HTMLElement>("#chart-tooltip");
  const picker = document.querySelector<HTMLElement>("#year-picker");
  const table = document.querySelector<HTMLElement>("#track-table-body");
  if (!chart || !tooltip || !picker || !table || !years.length) return;
  renderTable(table);
  let selected = years[years.length - 1];
  const select = (item: YearRecord) => {
    selected = item;
    picker.querySelectorAll("button").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.year === String(item.year)));
    });
    renderChart(chart, tooltip, item);
  };
  let renderedWidth = 0;
  new ResizeObserver(() => {
    if (Math.abs(chart.clientWidth - renderedWidth) < 8) return;
    renderedWidth = chart.clientWidth;
    renderChart(chart, tooltip, selected);
  }).observe(chart);
  for (const item of years) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.year = String(item.year);
    button.textContent = label(item.year);
    button.addEventListener("click", () => select(item));
    picker.append(button);
  }
  select(selected);
}

export function initDeadlineCountdown(): void {
  const target = document.querySelector<HTMLElement>("#deadline-countdown");
  if (!target) return;
  // 12 January 2027, 13:00 CET (UTC+1).
  const deadline = Date.UTC(2027, 0, 12, 12, 0, 0);
  const days = Math.ceil((deadline - Date.now()) / 86_400_000);
  target.textContent = days > 1 ? ` · ${days} days left` : days === 1 ? " · last day" : " · submission closed";
}
