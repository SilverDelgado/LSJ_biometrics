import argparse
from pathlib import Path

from lsj_biometrics.campaign.image import Image
from lsj_biometrics.modules.detectors import MTCNNFaceDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    image = Image(image_id=args.image_path.stem, path=args.image_path)
    result = MTCNNFaceDetector().detect(image)

    print(f"image={result.image.image_id}")
    print(f"faces={len(result.faces)}")
    for face in result.faces:
        print(f"{face.face_id}: box={face.bounding_box}, confidence={face.confidence:.6f}")


if __name__ == "__main__":
    main()
