"""딥러닝 얼굴 인식: 등록된 내 얼굴 O, 다른 얼굴 X.

설치 (Python 3.10 이상 권장):
    python -m pip install --upgrade "opencv-python>=4.10,<5" numpy

1. 본인 얼굴만 나오는 영상으로 등록 (한 번):
    python face_ox.py --enroll-video "20261110_170224.mp4"

2. 웹캠에서 실시간 판별:
    python face_ox.py

3. 다른 동영상에서 판별 (선택):
    python face_ox.py --input-video "test.mp4"

필요한 YuNet/SFace 딥러닝 모델은 최초 실행 시 OpenCV 공식 저장소에서 받습니다.
얼굴 특징 벡터(owner_face.npy)는 이 PC에만 저장됩니다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "패키지가 없습니다. 먼저 다음 명령어를 실행하세요:\n"
        'python -m pip install --upgrade "opencv-python>=4.10,<5" numpy'
    ) from exc


ROOT = Path(__file__).resolve().parent
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        100_000,
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        10_000_000,
    ),
}


def get_models(model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, (url, min_bytes) in MODELS.items():
        destination = model_dir / filename
        if destination.is_file() and destination.stat().st_size >= min_bytes:
            paths.append(destination)
            continue

        print(f"모델 다운로드 중: {filename}")
        temp_path = None
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=90) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=model_dir, suffix=".download", delete=False
                ) as output:
                    temp_path = Path(output.name)
                    shutil.copyfileobj(response, output)
            if temp_path.stat().st_size < min_bytes:
                raise RuntimeError("다운로드한 파일 크기가 예상보다 작습니다.")
            os.replace(temp_path, destination)
            paths.append(destination)
        except Exception as exc:
            raise RuntimeError(
                f"모델을 받을 수 없습니다: {filename}\n"
                f"OpenCV Zoo에서 직접 다운로드하여 {model_dir}에 넣으세요."
            ) from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return paths[0], paths[1]


def create_networks(model_dir: Path):
    if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError("OpenCV 버전이 낮습니다. opencv-python을 업그레이드하세요.")
    detection_path, recognition_path = get_models(model_dir)
    detector = cv2.FaceDetectorYN.create(
        str(detection_path), "", (320, 320), 0.85, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(recognition_path), "")
    return detector, recognizer


def shrink_frame(frame: np.ndarray, max_size: int = 640) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (round(width * scale), round(height * scale)))


def detect(detector, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return faces


def extract_feature(recognizer, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
    aligned_face = recognizer.alignCrop(frame, face)
    feature = recognizer.feature(aligned_face).reshape(-1).astype(np.float32)
    length = float(np.linalg.norm(feature))
    if length < 1e-8:
        raise RuntimeError("얼굴 특징 추출에 실패했습니다.")
    return feature / length


def enroll_from_video(video_path: Path, detector, recognizer, db_path: Path) -> None:
    if not video_path.is_file():
        raise RuntimeError(f"영상을 찾을 수 없습니다: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다. 필요하면 H.264 MP4로 변환하세요.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps * 0.35)) if np.isfinite(fps) and fps > 0 else 10
    features = []
    count = 0
    try:
        while len(features) < 30:
            ok, frame = cap.read()
            if not ok:
                break
            if count % step != 0:
                count += 1
                continue
            count += 1
            frame = shrink_frame(frame)
            faces = detect(detector, frame)
            # 등록 영상에는 본인 한 명만 나와야 합니다.
            if len(faces) != 1:
                continue
            if min(float(faces[0][2]), float(faces[0][3])) < 75:
                continue
            features.append(extract_feature(recognizer, frame, faces[0]))
    finally:
        cap.release()

    if len(features) < 5:
        raise RuntimeError(
            f"등록 가능한 얼굴 프레임이 {len(features)}개입니다. "
            "본인 얼굴 한 명이 선명하게 보이는 5초 이상 영상을 사용하세요."
        )

    # 여러 각도의 특징을 평균한 뒤 단위 벡터로 정규화합니다.
    owner = np.mean(np.stack(features), axis=0)
    owner /= np.linalg.norm(owner)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(db_path, owner.astype(np.float32))
    print(f"등록 완료: {len(features)}개 프레임 -> {db_path}")


def draw_result(frame: np.ndarray, face: np.ndarray, label: str, score: float):
    x, y, width, height = (int(v) for v in face[:4])
    color = (0, 210, 0) if label == "O" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.putText(
        frame, f"{label} ({score:.2f})", (max(0, x), max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA,
    )


def recognize(source, detector, recognizer, db_path: Path, threshold: float):
    if not db_path.is_file():
        raise RuntimeError("본인 얼굴이 등록되지 않았습니다. --enroll-video 를 먼저 실행하세요.")
    owner = np.load(db_path, allow_pickle=False).reshape(-1).astype(np.float32)
    owner_length = np.linalg.norm(owner)
    if owner.size == 0 or not np.isfinite(owner_length) or owner_length < 1e-8:
        raise RuntimeError("등록 파일이 올바르지 않습니다. 얼굴을 다시 등록하세요.")
    owner /= owner_length

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"카메라/영상을 열 수 없습니다: {source}")
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = shrink_frame(frame)
            faces = detect(detector, frame)
            for face in faces:
                feature = extract_feature(recognizer, frame, face)
                if feature.shape != owner.shape:
                    raise RuntimeError("등록된 특징 벡터 차원이 다릅니다. 다시 등록하세요.")
                score = float(np.dot(owner, feature))  # 코사인 유사도: -1~1
                label = "O" if score >= threshold else "X"
                draw_result(frame, face, label, score)

            if len(faces) == 0:
                cv2.putText(frame, "NO FACE", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, "Q / ESC: quit", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            cv2.imshow("Deep Learning Face O/X", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="내 얼굴은 O, 다른 얼굴은 X 표시")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enroll-video", type=Path, metavar="MP4",
                       help="본인 얼굴이 담긴 영상 등록")
    group.add_argument("--input-video", type=Path, metavar="MP4",
                       help="동영상을 판별 (생략 시 웹캠)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="본인 판정 코사인 유사도 임계값 (기본 0.45)")
    parser.add_argument("--db", type=Path, default=ROOT / "owner_face.npy")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold 값은 -1부터 1 사이여야 합니다.")

    try:
        detector, recognizer = create_networks(args.model_dir)
        if args.enroll_video is not None:
            enroll_from_video(args.enroll_video, detector, recognizer, args.db)
        else:
            source = str(args.input_video) if args.input_video else args.camera_index
            recognize(source, detector, recognizer, args.db, args.threshold)
    except (RuntimeError, cv2.error, OSError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
