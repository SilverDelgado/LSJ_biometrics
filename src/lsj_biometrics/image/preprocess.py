from lsj_biometrics.campaign.face import CroppedFaceImage, Face


class ImagePreprocessor:
    def crop(self, face: Face) -> CroppedFaceImage:
        raise NotImplementedError("Image cropping requires an image backend")