/* charts.js - draws charts with Chart.js using REAL data only:
   - dashboard charts: fetched from /api/dashboard-data
   - clustering map:   read from the JSON block rendered by the server */
(function () {
  "use strict";

  var COLORS = ["#2dd4bf", "#3b82f6", "#a78bfa", "#f59e0b", "#f472b6", "#34d399", "#60a5fa", "#fb7185", "#fbbf24", "#22d3ee"];

  function chartLibraryMissing() {
    document.querySelectorAll("canvas[data-dashboard-chart], #cluster-scatter").forEach(function (canvas) {
      var note = document.createElement("p");
      note.className = "muted";
      note.textContent = "Charts could not load (Chart.js is fetched from a CDN, so an internet connection is needed).";
      canvas.replaceWith(note);
    });
  }

  if (typeof Chart === "undefined") { chartLibraryMissing(); return; }
  Chart.defaults.color = "#9fb0cc";
  Chart.defaults.borderColor = "rgba(159,176,204,0.18)";
  Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";

  function find(name) { return document.querySelector("canvas[data-dashboard-chart='" + name + "']"); }

  function bar(canvas, labels, values, axisTitle) {
    new Chart(canvas, {
      type: "bar",
      data: { labels: labels, datasets: [{ data: values, backgroundColor: COLORS, borderRadius: 6 }] },
      options: { plugins: { legend: { display: false } },
                 scales: { y: { beginAtZero: true, ticks: { precision: 0 }, title: { display: true, text: axisTitle } } } }
    });
  }

  function roc(canvas, curves) {
    var datasets = Object.keys(curves).map(function (label, i) {
      var c = curves[label];
      return { label: label, showLine: true, pointRadius: 0, borderWidth: 2, borderColor: COLORS[i % COLORS.length],
               data: c.fpr.map(function (x, j) { return { x: x, y: c.tpr[j] }; }) };
    });
    datasets.push({ label: "Chance", showLine: true, pointRadius: 0, borderDash: [6, 6], borderColor: "#64748b",
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }] });
    new Chart(canvas, {
      type: "scatter", data: { datasets: datasets },
      options: { scales: { x: { min: 0, max: 1, title: { display: true, text: "False positive rate" } },
                           y: { min: 0, max: 1, title: { display: true, text: "True positive rate" } } } }
    });
  }

  // ---- Dashboard page -------------------------------------------------------
  if (document.querySelector("canvas[data-dashboard-chart]")) {
    fetch("/api/dashboard-data", { headers: { "Accept": "application/json" } })
      .then(function (r) { if (!r.ok) { throw new Error("HTTP " + r.status); } return r.json(); })
      .then(function (data) {
        var live = data.live, el;
        if ((el = find("credibility"))) {
          bar(el, Object.keys(live.credibility_labels), Object.values(live.credibility_labels), "Analyses");
        }
        if ((el = find("sentiment"))) {
          var keys = Object.keys(live.sentiment_labels);
          new Chart(el, { type: "doughnut",
            data: { labels: keys, datasets: [{ data: keys.map(function (k) { return live.sentiment_labels[k]; }),
                    backgroundColor: COLORS, borderWidth: 0 }] } });
        }
        if ((el = find("clusters")) && live.latest_clustering && live.latest_clustering.sizes) {
          var sizes = live.latest_clustering.sizes;
          bar(el, sizes.map(function (_, i) { return "Cluster " + (i + 1); }), sizes, "Texts");
        }
        ["credibility", "sentiment"].forEach(function (key) {
          var canvas = find("roc-" + key), report = data.offline[key];
          if (canvas && report && report.roc_curves) { roc(canvas, report.roc_curves); }
        });
      })
      .catch(function () {
        document.querySelectorAll("canvas[data-dashboard-chart]").forEach(function (canvas) {
          var note = document.createElement("p");
          note.className = "muted";
          note.textContent = "Could not load chart data. Refresh the page to try again.";
          canvas.replaceWith(note);
        });
      });
  }

  // ---- Clustering page ------------------------------------------------------
  var pointsEl = document.getElementById("cluster-points");
  var scatter = document.getElementById("cluster-scatter");
  if (pointsEl && scatter) {
    var points = JSON.parse(pointsEl.textContent), groups = {};
    points.forEach(function (p) { (groups[p.cluster] = groups[p.cluster] || []).push({ x: p.x, y: p.y }); });
    new Chart(scatter, {
      type: "scatter",
      data: { datasets: Object.keys(groups).map(function (c) {
        return { label: "Cluster " + (parseInt(c, 10) + 1), data: groups[c], backgroundColor: COLORS[c % COLORS.length], pointRadius: 5 };
      }) },
      options: { scales: { x: { title: { display: true, text: "PCA component 1" } }, y: { title: { display: true, text: "PCA component 2" } } } }
    });
  }
})();
