const METRICS = [
  ["ttft_p50_ms", "TTFT p50", "ms"],
  ["tpot_p50_ms", "TPOT p50", "ms"],
  ["e2e_p50_ms", "E2E p50", "ms"],
  ["output_tok_per_s", "Output tok/s", "tok/s"],
  ["goodput", "Goodput", ""],
];
const SERIES = ["#8c3a1e", "#b45309", "#1d6b45", "#1f4e68", "#6b3fa0"];

const state = {
  runs: [],
  current: "",
  detail: null,
  compareId: "",
  compare: null,
  metric: "ttft_p50_ms",
  output: null,
  highlight: "",
};

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") node.className = value;
    else if (key === "on" && value) {
      for (const [event, fn] of Object.entries(value)) node.addEventListener(event, fn);
    } else if (value != null && value !== false) node.setAttribute(key, String(value));
  }
  for (const child of children || []) {
    node.append(child instanceof Node ? child : document.createTextNode(child == null ? "" : String(child)));
  }
  return node;
}

function svg(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function fmtInt(value) {
  return value == null ? "—" : Number(value).toLocaleString();
}

function duration(seconds) {
  if (seconds == null) return "";
  const whole = Math.round(seconds);
  if (whole < 60) return `${whole}s`;
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

function when(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function deviceChip(device) {
  const name = device || "unset";
  return el("span", { class: `chip ${name}` }, [name]);
}

function metricMeta(key) {
  return METRICS.find((item) => item[0] === key) || METRICS[0];
}

function outputsOf(detail) {
  return [...new Set((detail.cells || []).map((cell) => cell.output_tokens))].sort((a, b) => a - b);
}

function cellsFor(detail, output) {
  return (detail.cells || []).filter((cell) => cell.output_tokens === output);
}

function goodputClass(value) {
  if (value == null) return "";
  if (value >= 0.999) return "good";
  if (value >= 0.5) return "mid";
  return "low";
}

function heatColor(value) {
  if (value == null) return "#efeae1";
  const stops = [[0, [244, 199, 195]], [0.5, [243, 221, 176]], [1, [207, 230, 214]]];
  const t = Math.min(1, Math.max(0, value));
  let lo = stops[0];
  let hi = stops[stops.length - 1];
  for (let i = 1; i < stops.length; i += 1) {
    if (t <= stops[i][0]) {
      lo = stops[i - 1];
      hi = stops[i];
      break;
    }
  }
  const span = hi[0] - lo[0] || 1;
  const mix = (t - lo[0]) / span;
  const rgb = lo[1].map((channel, i) => Math.round(channel + (hi[1][i] - channel) * mix));
  return `rgb(${rgb.join(",")})`;
}

function barFor(metric, bars) {
  if (!bars) return null;
  if (metric === "ttft_p50_ms") return bars.ttft_ms;
  if (metric === "tpot_p50_ms") return bars.tpot_ms;
  return null;
}

function formatY(metric, value) {
  if (metric === "goodput") return `${Math.round(value * 100)}%`;
  return String(Math.round(value));
}

function formatCell(metric, value) {
  if (value == null) return "—";
  if (metric === "goodput") return Number(value).toFixed(2);
  if (metric === "output_tok_per_s") return Number(value).toFixed(1);
  return String(Math.round(value));
}

function fails(cell) {
  const parts = [];
  if (cell.fail_timeout) parts.push(`${cell.fail_timeout} timeout`);
  if (cell.fail_context_length) parts.push(`${cell.fail_context_length} context`);
  if (cell.fail_oom) parts.push(`${cell.fail_oom} oom`);
  if (cell.fail_other) parts.push(`${cell.fail_other} other`);
  return parts.join(", ") || "0";
}

function seriesOf(detail, output, metric) {
  const rows = cellsFor(detail, output);
  const inputs = [...new Set(rows.map((cell) => cell.input_tokens))].sort((a, b) => a - b);
  return inputs.map((input, index) => ({
    input,
    color: SERIES[index % SERIES.length],
    points: rows
      .filter((cell) => cell.input_tokens === input && cell[metric] != null)
      .map((cell) => ({ x: cell.concurrency, y: cell[metric] }))
      .sort((a, b) => a.x - b.x),
  }));
}

function chartSvg(detail, compare, metric, output) {
  const width = 720;
  const height = 280;
  const pad = { l: 54, r: 12, t: 16, b: 32 };
  const series = seriesOf(detail, output, metric);
  const other = compare ? seriesOf(compare, output, metric) : [];
  const xs = [...new Set(series.flatMap((line) => line.points.map((point) => point.x)))].sort((a, b) => a - b);
  const values = series.flatMap((line) => line.points.map((point) => point.y));
  other.forEach((line) => line.points.forEach((point) => values.push(point.y)));
  const bar = barFor(metric, detail.bars);
  if (bar != null) values.push(bar);
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  let max = Math.max(1, ...values);
  if (metric === "goodput") max = 1;
  else max *= 1.08;
  const xAt = (value) => {
    if (xs.length <= 1) return pad.l + plotW / 2;
    const index = Math.max(0, xs.indexOf(value));
    return pad.l + (index / (xs.length - 1)) * plotW;
  };
  const yAt = (value) => pad.t + (1 - value / max) * plotH;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  const title = svg("title", {});
  title.textContent = `${metricMeta(metric)[1]} versus concurrent users`;
  root.append(title);
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = (max * tick) / 4;
    const y = yAt(value);
    root.append(svg("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, stroke: "#eee6da", "stroke-width": 1 }));
    const label = svg("text", {
      x: pad.l - 8, y: y + 4, "text-anchor": "end", fill: "#6f675c", "font-size": 11,
    });
    label.textContent = formatY(metric, value);
    root.append(label);
  }
  if (bar != null && bar <= max) {
    const y = yAt(bar);
    root.append(svg("line", {
      x1: pad.l, x2: width - pad.r, y1: y, y2: y,
      stroke: "#8c3a1e", "stroke-width": 1, "stroke-dasharray": "4 4",
    }));
  }
  function draw(lines, dashed) {
    for (const line of lines) {
      if (!line.points.length) continue;
      const d = line.points.map((point, i) => `${i ? "L" : "M"} ${xAt(point.x).toFixed(1)} ${yAt(point.y).toFixed(1)}`).join(" ");
      const path = svg("path", {
        d, fill: "none", stroke: line.color, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-linecap": "round",
      });
      if (dashed) path.setAttribute("stroke-dasharray", "5 4");
      root.append(path);
      for (const point of line.points) {
        root.append(svg("circle", {
          cx: xAt(point.x).toFixed(1), cy: yAt(point.y).toFixed(1), r: dashed ? 2.5 : 3.5,
          fill: dashed ? "#faf7f2" : line.color, stroke: line.color, "stroke-width": 1.5,
        }));
      }
    }
  }
  draw(series, false);
  draw(other, true);
  xs.forEach((value) => {
    const label = svg("text", {
      x: xAt(value), y: height - 10, "text-anchor": "middle", fill: "#6f675c", "font-size": 12,
    });
    label.textContent = String(value);
    root.append(label);
  });
  return { svg: root, series, bar };
}

function findCell(detail, concurrency, input, output) {
  return (detail.cells || []).find((cell) => (
    cell.concurrency === concurrency && cell.input_tokens === input && cell.output_tokens === output
  ));
}

function ratioFor(metric, primary, other) {
  const a = primary && primary[metric];
  const b = other && other[metric];
  if (a == null || b == null) return null;
  const higher = metric === "output_tok_per_s" || metric === "goodput";
  if (higher) {
    if (!b) return null;
    return a / b;
  }
  if (!a) return null;
  return b / a;
}

function renderRuns() {
  const box = document.getElementById("runs");
  box.replaceChildren();
  if (!state.runs.length) {
    box.append(el("p", { class: "empty" }, ["No summary.json files in results/ or examples/."]));
    return;
  }
  for (const run of state.runs) {
    const cells = run.requests ? `${fmtInt(run.requests)} requests` : "no cells";
    const button = el("button", {
      class: "run",
      type: "button",
      ...(run.id === state.current ? { "aria-current": "true" } : {}),
      on: { click: () => selectRun(run.id) },
    }, [
      el("span", { class: "who" }, [deviceChip(run.device), el("span", { class: "meta" }, [duration(run.duration_s)])]),
      el("span", { class: "model" }, [run.model || run.label]),
      el("span", { class: "meta" }, [cells]),
    ]);
    box.append(button);
  }
}

function renderMain() {
  const main = document.getElementById("main");
  const detail = state.detail;
  if (!detail) {
    main.replaceChildren(el("p", { class: "empty" }, ["Pick a run."]));
    return;
  }
  const outputs = outputsOf(detail);
  if (state.output == null || !outputs.includes(state.output)) state.output = outputs[0];
  const [, metricName, unit] = metricMeta(state.metric);
  const chart = chartSvg(detail, state.compare, state.metric, state.output);
  const legend = el("div", { class: "legend" }, chart.series.map((line) => (
    el("span", {}, [el("i", { class: "swatch", style: `background:${line.color}` }), `${fmtInt(line.input)} in`])
  )));
  if (state.compare) {
    legend.append(el("span", {}, [`Dashed: ${state.compare.model || state.compare.label}`]));
  }
  const captionBits = ["Each line is one input length. The horizontal axis is concurrent users."];
  if (chart.bar != null) captionBits.push(`The dashed rule is the ${chart.bar.toLocaleString()} ${unit || "ms"} bar.`);

  const head = el("div", { class: "head" }, [
    el("div", {}, [
      deviceChip(detail.device),
      el("h1", {}, [detail.model || detail.label]),
      el("p", { class: "sub" }, [
        [
          detail.dtype ? `dtype ${detail.dtype}` : null,
          detail.max_model_len ? `${fmtInt(detail.max_model_len)} context` : null,
          duration(detail.duration_s),
          when(detail.started_at),
          detail.client_cpus ? `client ${detail.client_cpus} CPUs` : null,
        ].filter(Boolean).join(" · "),
      ]),
    ]),
  ]);
  if (detail.endpoint) {
    head.querySelector(".sub").append(document.createTextNode(" · "), el("code", {}, [detail.endpoint]));
  }

  main.replaceChildren(
    head,
    kpis(detail),
    el("section", { class: "card" }, [
      el("h2", {}, [metricName]),
      toolbar(detail, outputs),
      el("div", { class: "chart" }, [chart.svg]),
      legend,
      el("p", { class: "caption" }, [captionBits.join(" ")]),
      compareTable(detail),
    ]),
    heatmap(detail),
    gridTable(detail),
    launchBlock(detail),
  );
}

function kpis(detail) {
  const pct = detail.requests ? Math.round((100 * detail.requests_ok) / detail.requests) : null;
  const peak = detail.peak;
  const broken = detail.break;
  const bars = detail.bars || {};
  const hasBar = bars.ttft_ms != null || bars.tpot_ms != null;
  let breakCard;
  if (!hasBar) {
    breakCard = kpi("Bar", "Unset", "This run recorded latency without a goodput bar.", "");
  } else if (!broken) {
    breakCard = kpi("Bar", "Held", `TTFT ${fmtInt(bars.ttft_ms)} ms and TPOT ${fmtInt(bars.tpot_ms)} ms on every request.`, "held");
  } else {
    breakCard = kpi(
      "Leaves the bar",
      `${broken.concurrency} × ${fmtInt(broken.input_tokens)}`,
      `${fmtInt(broken.output_tokens)} out · goodput ${Number(broken.goodput).toFixed(2)} · TTFT p99 ${fmtInt(broken.ttft_p99_ms)} ms`,
      "miss",
    );
  }
  return el("div", { class: "kpis" }, [
    kpi("Cells", String((detail.cells || []).length), `${fmtInt(detail.requests_ok)} / ${fmtInt(detail.requests)} requests ok`, ""),
    kpi("Succeeded", pct == null ? "—" : `${pct}%`, "Share of requests that finished.", ""),
    kpi(
      "Peak output",
      peak ? String(Math.round(peak.output_tok_per_s)) : "—",
      peak ? `tok/s at ${peak.concurrency} users, ${fmtInt(peak.input_tokens)} in, ${fmtInt(peak.output_tokens)} out` : "No throughput recorded.",
      "",
    ),
    breakCard,
  ]);
}

function kpi(label, value, hint, kind) {
  return el("article", { class: `kpi ${kind}` }, [
    el("div", { class: "label" }, [label]),
    el("div", { class: "value" }, [value]),
    el("div", { class: "hint" }, [hint]),
  ]);
}

function toolbar(detail, outputs) {
  const pills = el("div", { class: "pills", role: "group", "aria-label": "Metric" }, METRICS.map(([key, name]) => (
    el("button", {
      class: "pill",
      type: "button",
      "aria-pressed": key === state.metric ? "true" : "false",
      on: { click: () => { state.metric = key; renderMain(); } },
    }, [name])
  )));
  const outs = el("div", { class: "pills", role: "group", "aria-label": "Output length" }, outputs.map((output) => (
    el("button", {
      class: "pill",
      type: "button",
      "aria-pressed": output === state.output ? "true" : "false",
      on: { click: () => { state.output = output; state.highlight = ""; renderMain(); } },
    }, [`${fmtInt(output)} out`])
  )));
  const others = state.runs.filter((run) => run.id !== detail.id);
  const compare = el("label", {}, ["Compare"]);
  const select = el("select", {
    "aria-label": "Compare with another run",
    on: { change: (event) => selectCompare(event.target.value) },
  }, [el("option", { value: "" }, ["None"])]);
  for (const run of others) {
    const option = el("option", { value: run.id }, [`${run.device || "run"} · ${run.model || run.label}`]);
    if (run.id === state.compareId) option.selected = true;
    select.append(option);
  }
  compare.append(select);
  const row = el("div", { class: "toolbar" }, [pills, outs]);
  if (others.length) row.append(compare);
  return row;
}

function heatmap(detail) {
  const rows = cellsFor(detail, state.output);
  const users = [...new Set(rows.map((cell) => cell.concurrency))].sort((a, b) => a - b);
  const inputs = [...new Set(rows.map((cell) => cell.input_tokens))].sort((a, b) => a - b);
  const table = el("table", { class: "heat" }, [
    el("thead", {}, [el("tr", {}, [
      el("th", {}, ["users"]),
      ...inputs.map((input) => el("th", {}, [`${fmtInt(input)} in`])),
    ])]),
  ]);
  const body = el("tbody");
  for (const user of users) {
    const tr = el("tr", {}, [el("th", { class: "row" }, [String(user)])]);
    for (const input of inputs) {
      const cell = rows.find((item) => item.concurrency === user && item.input_tokens === input);
      const key = cell ? `${cell.concurrency}-${cell.input_tokens}-${cell.output_tokens}` : "";
      const value = cell ? cell.goodput : null;
      const button = el("button", {
        class: "heat-cell",
        type: "button",
        style: `background:${heatColor(value)}`,
        "aria-pressed": key && key === state.highlight ? "true" : "false",
        on: { click: () => { state.highlight = key; renderMain(); } },
      }, [value == null ? "—" : Number(value).toFixed(2)]);
      tr.append(el("td", {}, [button]));
    }
    body.append(tr);
  }
  table.append(body);
  return el("section", { class: "card" }, [
    el("h2", {}, [`Goodput · ${fmtInt(state.output)} out`]),
    el("p", { class: "caption" }, ["Share of requests under both bars. A cell at 1.00 held. Click a cell to mark its row."]),
    table,
  ]);
}

function gridTable(detail) {
  const rows = cellsFor(detail, state.output);
  const head = ["C", "in", "out", "ok", "TTFT p50", "TTFT p99", "TPOT p50", "E2E p50", "tok/s", "goodput", "fails"];
  const table = el("table", { class: "grid" }, [
    el("thead", {}, [el("tr", {}, head.map((name) => el("th", {}, [name])))]),
  ]);
  const body = el("tbody");
  for (const cell of rows) {
    const key = `${cell.concurrency}-${cell.input_tokens}-${cell.output_tokens}`;
    const good = el("td", { class: goodputClass(cell.goodput) }, [cell.goodput == null ? "—" : Number(cell.goodput).toFixed(2)]);
    body.append(el("tr", { class: key === state.highlight ? "is-hot" : "", id: `cell-${key}` }, [
      el("td", {}, [String(cell.concurrency)]),
      el("td", {}, [fmtInt(cell.input_tokens)]),
      el("td", {}, [fmtInt(cell.output_tokens)]),
      el("td", {}, [`${cell.ok}/${cell.requests}`]),
      el("td", {}, [fmtInt(cell.ttft_p50_ms)]),
      el("td", {}, [fmtInt(cell.ttft_p99_ms)]),
      el("td", {}, [fmtInt(cell.tpot_p50_ms)]),
      el("td", {}, [fmtInt(cell.e2e_p50_ms)]),
      el("td", {}, [cell.output_tok_per_s == null ? "—" : Number(cell.output_tok_per_s).toFixed(1)]),
      good,
      el("td", {}, [fails(cell)]),
    ]));
  }
  table.append(body);
  return el("section", { class: "card" }, [
    el("h2", {}, ["Cells"]),
    el("div", { class: "table-wrap" }, [table]),
  ]);
}

function compareTable(detail) {
  const other = state.compare;
  if (!other) return el("span");
  const gpu = detail.device === "gpu" ? detail : other.device === "gpu" ? other : null;
  const cpu = detail.device === "cpu" ? detail : other.device === "cpu" ? other : null;
  const paired = gpu && cpu && gpu !== cpu;
  const left = paired ? gpu : detail;
  const right = paired ? cpu : other;
  const higher = state.metric === "output_tok_per_s" || state.metric === "goodput";
  const ratioName = paired ? (higher ? "GPU/CPU" : "CPU/GPU") : "ratio";
  const [, metricName] = metricMeta(state.metric);
  const head = el("tr", {}, ["C", "in", "out", left.device || "selected", right.device || "compare", ratioName].map((name) => el("th", {}, [name])));
  const body = el("tbody");
  for (const cell of cellsFor(left, state.output)) {
    const match = findCell(right, cell.concurrency, cell.input_tokens, cell.output_tokens);
    const ratio = ratioFor(state.metric, paired ? findCell(gpu, cell.concurrency, cell.input_tokens, cell.output_tokens) : cell, paired ? findCell(cpu, cell.concurrency, cell.input_tokens, cell.output_tokens) : match);
    body.append(el("tr", {}, [
      el("td", {}, [String(cell.concurrency)]),
      el("td", {}, [fmtInt(cell.input_tokens)]),
      el("td", {}, [fmtInt(cell.output_tokens)]),
      el("td", {}, [formatCell(state.metric, (paired ? findCell(left, cell.concurrency, cell.input_tokens, cell.output_tokens) : cell)[state.metric])]),
      el("td", {}, [formatCell(state.metric, match && match[state.metric])]),
      el("td", {}, [ratio == null ? "—" : `${ratio.toFixed(2)}×`]),
    ]));
  }
  return el("div", {}, [
    el("h2", { style: "margin-top:18px" }, [`${metricName} side by side`]),
    el("p", { class: "caption" }, [paired
      ? (higher ? "Ratio is GPU divided by CPU." : "Ratio is CPU divided by GPU.")
      : "Ratio is the right column divided by the left for latency, and the left divided by the right for throughput."]),
    el("div", { class: "table-wrap" }, [el("table", { class: "grid" }, [el("thead", {}, [head]), body])]),
  ]);
}

function launchBlock(detail) {
  if (!detail.launch_args) return el("span");
  return el("details", { class: "launch" }, [
    el("summary", {}, ["Launch flags"]),
    el("pre", {}, [detail.launch_args]),
  ]);
}

async function fetchRun(id) {
  const response = await fetch(`/api/runs/${id}`);
  if (!response.ok) throw new Error(`run ${id} returned ${response.status}`);
  return response.json();
}

async function selectRun(id) {
  state.current = id;
  location.hash = id;
  renderRuns();
  document.getElementById("status").textContent = "Loading the run";
  state.detail = await fetchRun(id);
  const outputs = outputsOf(state.detail);
  const broken = state.detail.break;
  state.output = broken && outputs.includes(broken.output_tokens) ? broken.output_tokens : outputs[0];
  state.highlight = broken ? `${broken.concurrency}-${broken.input_tokens}-${broken.output_tokens}` : "";
  if (state.compareId === id) {
    state.compareId = "";
    state.compare = null;
  }
  document.getElementById("status").textContent = `${state.runs.length} run${state.runs.length === 1 ? "" : "s"} · localhost`;
  renderMain();
}

async function selectCompare(id) {
  state.compareId = id;
  state.compare = id ? await fetchRun(id) : null;
  renderMain();
}

async function boot() {
  const status = document.getElementById("status");
  try {
    const response = await fetch("/api/runs");
    state.runs = await response.json();
  } catch (error) {
    status.textContent = "The run list did not load.";
    document.getElementById("main").replaceChildren(el("p", { class: "empty" }, [String(error)]));
    return;
  }
  renderRuns();
  if (!state.runs.length) {
    status.textContent = "No runs yet";
    document.getElementById("main").replaceChildren(el("p", { class: "empty" }, [
      "Run a sweep, or keep a sanitized summary.json under examples/. Then reload this page.",
    ]));
    return;
  }
  const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
  const picked = state.runs.find((run) => run.id === hash) || state.runs[0];
  await selectRun(picked.id);
}

boot();
