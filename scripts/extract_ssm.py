import argparse
from pathlib import Path

from lsj_biometrics.campaign.image import Image
from lsj_biometrics.image.preprocess import ImagePreprocessor
from lsj_biometrics.modules.detectors import MTCNNFaceDetector
from lsj_biometrics.modules.extractors import SSMFeatureExtractor
from lsj_biometrics.modules.extractors.ssm import load_ssm_face_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/crops"))
    args = parser.parse_args()

    image = Image(image_id=args.image_path.stem, path=args.image_path)
    detection = MTCNNFaceDetector().detect(image)
    preprocessor = ImagePreprocessor(args.crops_dir)
    model = load_ssm_face_model(args.model) if args.model else None
    extractor = SSMFeatureExtractor(model, device=args.device)

    for face in detection.faces:
        template = extractor.extract(preprocessor.crop(face))
        print(f"{face.face_id}: dimension={template.dimension}, model={template.model_name}:{template.model_version}")


if __name__ == "__main__":
    main()
