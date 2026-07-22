"""The live training dashboard: a standalone HTTP server plus the client that
feeds it.

The server (``run_server`` / ``python -m babygrad.viz.serve``) is long-lived and
starts empty. A training run connects to it with ``connect`` — which POSTs the
run's graph and then streams each finished epoch to ``/ingest/*`` — so one
dashboard survives across many runs: starting a new run clears the board and
streams fresh. The browser reads GET payloads plus an SSE stream (``/events``);
none of that changes when the data source moved from a shared in-process
reference to cross-process POSTs.

Runs on ``ThreadingHTTPServer`` so the SSE stream and the ingest POSTs each get
their own thread. Server state lives on the server instance behind a lock, read
by the handler via ``self.server``.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

_STATIC_DIR = Path(__file__).parent / "static"

# How often the SSE loop and the push thread poll for newly recorded epochs.
_POLL_SECONDS = 0.2

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}

# The graph payload before any run has connected, so the frontend renders a
# "waiting" state rather than erroring on missing keys.
_EMPTY_GRAPH = {"nodes": [], "edges": [], "scopes": []}


class DashboardServer(ThreadingHTTPServer):
    """A threaded server holding the current run's payloads behind a lock.

    Starts empty and is filled by ``start_run``/``add_epoch`` as a training run
    connects and streams. ``generation`` counts runs: it bumps on every
    ``start_run`` so an open SSE stream can tell that a new run replaced the
    board and signal the browser to reload.
    """

    def __init__(self, address: tuple[str, int]):
        super().__init__(address, _DashboardHandler)
        self._lock = threading.Lock()
        self.graph_json: dict = _EMPTY_GRAPH
        self.node_stats: dict = {}
        self._history: dict = {}
        self.generation = 0

    def start_run(self, graph_json: dict, node_stats: dict) -> None:
        """Adopt a new run: swap in its graph, clear the history, bump the
        generation so open streams reset the browser onto this run."""
        with self._lock:
            self.graph_json = graph_json
            self.node_stats = node_stats
            self._history = {}
            self.generation += 1

    def add_epoch(self, delta: dict) -> None:
        """Merge one streamed epoch's scalars and series into the history."""
        step = delta["step"]
        with self._lock:
            for tag, value in {**delta["scalars"], **delta["series"]}.items():
                self._history.setdefault(tag, {})[step] = value

    def snapshot(self) -> tuple[dict, int]:
        """A consistent copy of the history and the current generation together,
        so a reader never pairs one run's history with another's generation."""
        with self._lock:
            history = {tag: dict(points) for tag, points in self._history.items()}
            return history, self.generation


class _DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        """Serve the JSON payloads and the SSE stream, else a static file. The
        query string (e.g. ?tab=plots) is ignored for routing."""
        server = cast("DashboardServer", self.server)
        route = self.path.split("?", 1)[0]
        if route == "/graph.json":
            self._send_json(server.graph_json)
        elif route == "/history.json":
            self._send_json(server.snapshot()[0])
        elif route == "/node_stats.json":
            self._send_json(server.node_stats)
        elif route == "/events":
            self._stream_events(server)
        else:
            # /theme.json and the frontend files are served from static/
            self._send_static(route)

    def do_POST(self) -> None:
        """Ingest from a connected training run: a new run's graph, or one epoch."""
        server = cast("DashboardServer", self.server)
        route = self.path.split("?", 1)[0]
        body = self._read_body()
        if route == "/ingest/run":
            server.start_run(body["graph"], body["node_stats"])
            self._send(204, "text/plain; charset=utf-8", b"")
        elif route == "/ingest/epoch":
            server.add_epoch(body)
            self._send(204, "text/plain; charset=utf-8", b"")
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def _read_body(self) -> dict:
        """Read exactly Content-Length bytes (leaving the connection framed for
        the next request) and parse the JSON body."""
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else {}

    def _stream_events(self, server: DashboardServer) -> None:
        """Stream one SSE per newly recorded epoch until the run changes or the
        client disconnects. Each poll re-snapshots; if the generation moved a new
        run replaced the board, so emit a ``reset`` (the browser reloads onto the
        new run) and end this stream. Its own thread, so blocking here is fine."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        _, opened_generation = server.snapshot()
        last_step = -1
        try:
            while True:
                history, generation = server.snapshot()
                if generation != opened_generation:
                    self._write_event("reset", {"generation": generation})
                    return
                for step in _steps_after(history, last_step):
                    self._write_event("epoch", _delta_for_step(history, step))
                    last_step = step
                time.sleep(_POLL_SECONDS)
        except (BrokenPipeError, ConnectionResetError):
            return  # client closed the tab; end this stream thread

    def _write_event(self, event: str, data: dict) -> None:
        """Write one SSE frame and flush so the browser sees it immediately."""
        frame = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    def _send_json(self, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(200, "application/json; charset=utf-8", body)

    def _send_static(self, path: str) -> None:
        """Serve a file from the static dir. ``/`` maps to index.html; anything that
        escapes the static dir or is missing is a 404."""
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (_STATIC_DIR / relative).resolve()
        if _STATIC_DIR.resolve() not in target.parents or not target.is_file():
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        content_type = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(200, content_type, target.read_bytes())

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        """Silence the default per-request stderr logging; the server prints its
        own one-line banner instead."""


def _steps_after(snapshot: dict, last_step: int) -> list[int]:
    """The recorded steps newer than ``last_step``, in order, across all series."""
    steps = {step for points in snapshot.values() for step in points}
    return sorted(step for step in steps if step > last_step)


def _delta_for_step(snapshot: dict, step: int) -> dict:
    """One epoch's delta: scalar values (loss, val_loss, metrics) split from
    distribution series (per-layer weights/grads), so the frontend can route each
    to the right chart."""
    scalars: dict = {}
    series: dict = {}
    for tag, points in snapshot.items():
        if step not in points:
            continue
        value = points[step]
        if isinstance(value, list):
            series[tag] = value
        else:
            scalars[tag] = value
    return {"step": step, "scalars": scalars, "series": series}


def _graph_snapshot(model, sample_input) -> tuple[dict, dict]:
    """One traced forward on ``sample_input`` → the static graph payload and a
    per-node value snapshot."""
    from babygrad.observers import Tracer
    from babygrad.tracing import tracing
    from babygrad.viz.attribution import attribute
    from babygrad.viz.graph import GraphVisualizer
    from babygrad.viz.graph_json import to_graph_json
    from babygrad.viz.node_stats import to_node_stats

    tracer = Tracer()
    with tracing(tracer):
        output = model.forward(sample_input)
    trace = attribute(tracer.records)
    payload = to_graph_json(GraphVisualizer(output, trace).graph, trace.scopes)
    return payload, to_node_stats(output)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the standalone dashboard server and serve until interrupted. Open the
    printed URL, then start a training run with ``--dashboard`` to stream to it."""
    server = DashboardServer((host, port))
    print(f"babygrad dashboard: http://{host}:{port} — start a --dashboard run to stream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class _Pusher:
    """Streams a live recorder's finished epochs to a running dashboard server.

    Polls the recorder from a daemon thread and POSTs each *completed* epoch — a
    step below ``recorder.step``, which is the epoch currently being written, so a
    poll never ships a half-recorded epoch. ``flush`` lifts that bound to push the
    final epoch, which the loop leaves behind since ``step`` never advances past it.
    """

    def __init__(self, recorder, base_url: str):
        self._recorder = recorder
        self._base_url = base_url
        self._last_step = -1
        self._lock = threading.Lock()  # serialises the poll thread with flush()
        self._stop = threading.Event()
        thread = threading.Thread(target=self._run, daemon=True, name="dashboard-push")
        thread.start()

    def _run(self) -> None:
        while not self._stop.wait(_POLL_SECONDS):
            self._push_below(self._recorder.step)

    def flush(self) -> None:
        """Stop polling and push whatever completed epochs remain, including the
        final one (``recorder.step`` itself, which the loop's bound excludes)."""
        self._stop.set()
        self._push_below(self._recorder.step + 1)

    def _push_below(self, boundary: int) -> None:
        """POST each not-yet-sent step strictly below ``boundary``, in order."""
        snapshot = self._recorder.snapshot()
        with self._lock:
            for step in _steps_after(snapshot, self._last_step):
                if step >= boundary:
                    break  # steps are sorted; the rest are still in progress
                _post_json(f"{self._base_url}/ingest/epoch", _delta_for_step(snapshot, step))
                self._last_step = step


def connect(
    model,
    recorder,
    sample_input,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> _Pusher | None:
    """Connect this training run to a running dashboard server and start streaming.

    Traces one forward to build the graph, POSTs it (which clears the board for
    this run), then returns a ``_Pusher`` streaming each finished epoch. Returns
    None if no server is reachable — training then proceeds without a dashboard.
    Call before ``fit``; call ``.flush()`` on the result after training so the
    final epoch lands.
    """
    base_url = f"http://{host}:{port}"
    payload, node_stats = _graph_snapshot(model, sample_input)
    try:
        _post_json(f"{base_url}/ingest/run", {"graph": payload, "node_stats": node_stats})
    except OSError:
        print(f"no dashboard server at {base_url} — start it with `just dashboard`")
        return None
    print(f"streaming to babygrad dashboard: {base_url}")
    return _Pusher(recorder, base_url)


def _post_json(url: str, payload: dict) -> None:
    """POST a JSON body, discarding the response. Connection errors and timeouts
    after the initial handshake are swallowed so a dead or hung dashboard can't
    crash training; only the initial connect re-raises (caught by ``connect``)."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(request, timeout=5).close()
    except OSError:
        if url.endswith("/ingest/run"):
            raise  # the initial connect must surface a missing server


if __name__ == "__main__":
    run_server()
