from lsj_biometrics.campaign.face import Face, FaceImage


class ImagePreprocessor:
    def crop(self, face: Face) -> FaceImage:
        raise NotImplementedError("Image cropping requires an image backend")