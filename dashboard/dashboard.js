const text = (id, value) => { document.getElementById(id).textContent = value; };

async function loadBenchmark() {
  try {
    const response = await fetch("../results/benchmark.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    render(data);
  } catch (error) {
    document.getElementById("status").textContent = "Could not load benchmark.json";
    document.querySelector(".metrics").setAttribute("title", error.message);
  }
}

function chartConfig(canvasId, chartType, labels, values, label, color) {
  return new Chart(document.getElementById(canvasId), {
    type: chartType,
    data: { labels, datasets: [{ label, data: values, borderColor: color, backgroundColor: color + "33", tension: .25, fill: true }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#99a4b8" } } },
      scales: {
        x: { ticks: { color:"#99a4b8" }, grid: { color:"#263044" } },
        y: { beginAtZero:true, ticks: { color:"#99a4b8" }, grid: { color:"#263044" } }
      }
    }
  });
}

function render(data) {
  const { summary, runs, metadata } = data;
  const label = data.metadata?.data_type === "sample" ? "Sample/demo data" : "Measured data";
  document.getElementById("status").textContent = `${label} · ${metadata.model_size || metadata.model}`;
  text("throughput", summary.throughput_tokens_per_sec?.toFixed(1) ?? "N/A");
  text("latency", summary.latency_ms_per_token?.toFixed(2) ?? "N/A");
  text("memory", summary.peak_gpu_memory_mb?.toFixed(0) ?? "N/A");

  const benchmarkLabels = data.benchmarks?.map(b => b.label) ?? (runs?.map((_, i) => `Run ${i + 1}`) ?? []);
  const throughputValues = data.benchmarks?.map(b => b.throughput_tokens_per_sec) ?? runs.map(r => r.throughput_tokens_per_sec);
  const memoryValues = data.benchmarks?.map(b => b.peak_gpu_memory_mb) ?? [summary.peak_gpu_memory_mb ?? 0];
  chartConfig("throughputChart", "bar", benchmarkLabels, throughputValues, "Tokens/sec", "#7c5cff");

  if (data.benchmarks) {
    chartConfig("latencyChart", "line", benchmarkLabels, data.benchmarks.map(b => b.latency_p50_ms), "p50 latency (ms/token)", "#38bdf8");
  } else {
    chartConfig("latencyChart", "bar", ["p50", "p90", "p99"], [summary.latency_p50_ms, summary.latency_p90_ms, summary.latency_p99_ms], "Latency (ms/token)", "#38bdf8");
  }
  chartConfig("memoryChart", "bar", data.benchmarks ? benchmarkLabels : [metadata.model_size || metadata.model], memoryValues, "Peak GPU memory (MiB)", "#34d399");
}

loadBenchmark();
