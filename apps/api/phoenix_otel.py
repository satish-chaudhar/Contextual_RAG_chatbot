# apps/api/phoenix_otel.py
import os
import logging
from typing import Optional

# OpenTelemetry core
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:
    trace = None  # graceful fallback
    TracerProvider = None  # type: ignore
    Resource = None  # type: ignore

# Try importing OTLP exporters (gRPC preferred, HTTP fallback)
OTLP_GRPC = None
OTLP_HTTP = None
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcSpanExporter
    OTLP_GRPC = OTLPGrpcSpanExporter
except Exception:
    OTLP_GRPC = None

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPRestSpanExporter
    OTLP_HTTP = OTLPRestSpanExporter
except Exception:
    OTLP_HTTP = None

_LOG = logging.getLogger("phoenix_otel")
_LOG.setLevel(logging.INFO)


def _resolve_endpoint() -> str:
    # Respect the environment variables you already have (docker-compose had these)
    # Prefer OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, then PHOENIX_OTLP_ENDPOINT, then PHOENIX_ENDPOINT
    for k in ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "PHOENIX_OTLP_ENDPOINT", "PHOENIX_ENDPOINT"):
        v = os.getenv(k)
        if v:
            return v
    # default to Phoenix OTLP gRPC collector inside compose
    return os.getenv("PHOENIX_DEFAULT_OTLP", "phoenix:4317")


def register(endpoint: Optional[str] = None, service_name: Optional[str] = None):
    """
    Register and configure OpenTelemetry tracing to send spans to Phoenix.
    - If an HTTP-style endpoint (starts with http) is provided, the HTTP OTLP exporter is used.
    - Otherwise the gRPC OTLP exporter is attempted.
    - If the exporter/instrumentation packages are not present, this function fails gracefully.
    Returns the TracerProvider or None if registration wasn't possible.
    """
    if trace is None or TracerProvider is None or Resource is None:
        _LOG.warning("OpenTelemetry SDK not available. Skipping OTEL registration.")
        return None

    endpoint = endpoint or _resolve_endpoint()
    service_name = service_name or os.getenv("PHOENIX_PROJECT", os.getenv("SERVICE_NAME", "openwebui-contextual-rag"))

    _LOG.info("Registering OTEL tracing to Phoenix. endpoint=%s service_name=%s", endpoint, service_name)

    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.getenv("SERVICE_VERSION", "0.0.1"),
        "deployment.environment": os.getenv("ENV", "dev"),
    })

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Choose exporter
    exporter = None
    try:
        if str(endpoint).lower().startswith("http"):
            if OTLP_HTTP is None:
                _LOG.warning("OTLP HTTP exporter not available. Can't use HTTP endpoint %s", endpoint)
            else:
                exporter = OTLP_HTTP(endpoint=endpoint)
                _LOG.info("Using OTLP HTTP exporter -> %s", endpoint)
        else:
            if OTLP_GRPC is None:
                _LOG.warning("OTLP gRPC exporter not available. Can't use gRPC endpoint %s", endpoint)
            else:
                # for proto.grpc exporter: use insecure channel when connecting inside docker-compose
                try:
                    exporter = OTLP_GRPC(endpoint=endpoint, insecure=True)
                except TypeError:
                    # Older/newer versions may have different signature
                    exporter = OTLP_GRPC(endpoint=endpoint)
                _LOG.info("Using OTLP gRPC exporter -> %s", endpoint)
    except Exception as e:
        _LOG.exception("Failed to create OTLP exporter: %s", e)

    if exporter is None:
        _LOG.warning("No OTLP exporter configured. Tracing is effectively disabled.")
        return provider

    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    # Optional automatic instrumentations if present (best-effort; do not crash)
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
        _LOG.info("Instrumented requests library for automatic spans.")
    except Exception:
        _LOG.debug("Requests instrumentation not available, continuing without it.", exc_info=True)

    # FastAPI instrumentation typically requires access to the app object and thus should be applied
    # from main.py with FastAPIInstrumentor.instrument_app(app). We won't call it here.

    return provider
