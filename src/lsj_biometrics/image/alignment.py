import cv2
import numpy as np

from lsj_biometrics.campaign.face import FaceLayout, Landmarks


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = source_centered.var(axis=0).sum()
    if not np.isfinite(variance) or variance <= np.finfo(np.float64).eps:
        raise ValueError("landmarks are degenerate")
    u, singular, vt = np.linalg.svd(target_centered.T @ source_centered / len(source))
    sign = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1] = -1
    rotation = u @ np.diag(sign) @ vt
    scale = (singular * sign).sum() / variance
    translation = target_mean - scale * rotation @ source_mean
    return np.hstack([scale * rotation, translation[:, None]])


def align_face(pixels: np.ndarray, landmarks: Landmarks, layout: FaceLayout) -> np.ndarray:
    matrix = similarity_transform(np.asarray(landmarks.points()), np.asarray(layout.landmarks.points()))
    return cv2.warpAffine(pixels, matrix, (layout.size, layout.size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
