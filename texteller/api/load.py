from pathlib import Path

import wget
from onnxruntime import InferenceSession
from transformers import RobertaTokenizerFast

from texteller.constants import LATEX_DET_MODEL_URL, TEXT_DET_MODEL_URL, TEXT_REC_MODEL_URL
from texteller.globals import Globals
from texteller.logger import get_logger
from texteller.models import TexTeller
from texteller.paddleocr import predict_det, predict_rec
from texteller.paddleocr.utility import parse_args
from texteller.utils import cuda_available, mkdir, resolve_path
from texteller.types import TexTellerModel

_logger = get_logger(__name__)


def load_model(model_dir: str | None = None, use_onnx: bool = False) -> TexTellerModel:
    return TexTeller.from_pretrained(model_dir, use_onnx=use_onnx)


def load_tokenizer(tokenizer_dir: str | None = None) -> RobertaTokenizerFast:
    return TexTeller.get_tokenizer(tokenizer_dir)


def load_latexdet_model() -> InferenceSession:
    fpath = _maybe_download(LATEX_DET_MODEL_URL)
    return InferenceSession(
        resolve_path(fpath),
        providers=["CUDAExecutionProvider" if cuda_available() else "CPUExecutionProvider"],
    )


def load_textrec_model() -> predict_rec.TextRecognizer:
    fpath = _maybe_download(TEXT_REC_MODEL_URL)
    paddleocr_args = parse_args()
    paddleocr_args.use_onnx = True
    paddleocr_args.rec_model_dir = resolve_path(fpath)
    paddleocr_args.use_gpu = cuda_available()
    predictor = predict_rec.TextRecognizer(paddleocr_args)
    return predictor


def load_textdet_model() -> predict_det.TextDetector:
    fpath = _maybe_download(TEXT_DET_MODEL_URL)
    paddleocr_args = parse_args()
    paddleocr_args.use_onnx = True
    paddleocr_args.det_model_dir = resolve_path(fpath)
    paddleocr_args.use_gpu = cuda_available()
    predictor = predict_det.TextDetector(paddleocr_args)
    return predictor


def _maybe_download(url: str, dirpath: str | None = None, force: bool = False) -> Path:
    if dirpath is None:
        dirpath = Globals().cache_dir
    mkdir(dirpath)

    fname = Path(url).name
    fpath = Path(dirpath) / fname
    if not fpath.exists() or force:
        _logger.info(f"Downloading {fname} from {url} to {fpath}")
        wget.download(url, resolve_path(fpath))

    return fpath
