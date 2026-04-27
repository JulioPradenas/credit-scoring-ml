import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.predictor import predictor, MODEL_VERSION
from api.schemas import (
    ApplicantInput, PredictionResponse,
    BatchInput, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model and scorecard...")
    predictor.load()
    logger.info(f"Champion model: {predictor.champion} | Version: {MODEL_VERSION}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crediticio con scorecard bancaria y modelo de ML.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {elapsed_ms:.1f}ms")
    return response


@app.get("/health", response_model=HealthResponse, tags=["Infra"])
def health() -> HealthResponse:
    if not predictor._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok", version=MODEL_VERSION)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Infra"])
def model_info() -> ModelInfoResponse:
    if not predictor._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(**predictor.model_info)


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(applicant: ApplicantInput) -> PredictionResponse:
    if not predictor._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predictor.predict_one(applicant)
    except Exception as exc:
        logger.error(f"Prediction error for {applicant.applicant_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Scoring"])
def predict_batch(payload: BatchInput) -> BatchPredictionResponse:
    if not predictor._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = predictor.predict_batch(payload.applicants)
        return BatchPredictionResponse(total=len(results), results=results)
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
