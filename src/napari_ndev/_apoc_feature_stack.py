from typing import TYPE_CHECKING

from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    LineEdit,
    ProgressBar,
    PushButton,
    TextEdit,
)
from napari import layers

if TYPE_CHECKING:
    import napari


class ApocFeatureStack(Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer" = None,
    ):
        super().__init__()
        self
        self._viewer = viewer if viewer is not None else None

        self._original = CheckBox(label="Keep Original Image")
        self._gaussian_blur = LineEdit(label="Gaussian Blur")
        self._DoG = LineEdit(label="Difference of Gauss.")
        self._LoG = LineEdit(label="Laplacian of Gauss.")
        self._SoG = LineEdit(label="Sobel of Gauss.")
        self._sHoG = LineEdit(label="Small Hessian of Gauss.")
        self._lHoG = LineEdit(label="Large Hessian of Gauss.")
        self._median = LineEdit(label="Median")
        self._tophat = LineEdit(label="Top Hat")

        self._generate_string_button = PushButton(
            label="Generate Feature String"
        )
        self._feature_string = TextEdit(label="Custom Feature String")

        self._image_layer = ComboBox(
            choices=self._filter_layers(layers.Image), label="Image Layer"
        )
        self._apply_button = PushButton(label="Apply to selected image")
        self._progress_bar = ProgressBar(label="Progress: ")

        self.extend(
            [
                self._original,
                self._gaussian_blur,
                self._DoG,
                self._LoG,
                self._SoG,
                self._sHoG,
                self._lHoG,
                self._median,
                self._tophat,
                self._generate_string_button,
                self._feature_string,
                self._image_layer,
                self._apply_button,
                self._progress_bar,
            ]
        )

        self._generate_string_button.clicked.connect(
            self.generate_feature_string
        )
        self._apply_button.clicked.connect(self.layer_to_feature_stack)

        if self._viewer is not None:
            self._viewer.layers.events.removed.connect(
                self._update_layer_choices
            )
            self._viewer.layers.events.inserted.connect(
                self._update_layer_choices
            )

    def _filter_layers(self, layer_type):
        if self._viewer is None:
            return []
        return [x for x in self._viewer.layers if isinstance(x, layer_type)]

    def _update_layer_choices(self):
        self._image_layer.choices = self._filter_layers(layers.Image)

    def generate_feature_string(self):
        def process_feature(prefix, input_str):
            return [
                prefix + num.strip()
                for num in input_str.split(",")
                if num.strip()
            ]

        feature_list = []
        if self._original.value:
            feature_list.append("original")
        feature_list.extend(
            process_feature("gaussian_blur=", self._gaussian_blur.value)
        )
        feature_list.extend(
            process_feature("difference_of_gaussian=", self._DoG.value)
        )
        feature_list.extend(
            process_feature("laplace_box_of_gaussian_blur=", self._LoG.value)
        )
        feature_list.extend(
            process_feature("sobel_of_gaussian_blur=", self._SoG.value)
        )
        feature_list.extend(
            process_feature(
                "small_hessian_eigenvalue_of_gaussian_blur=", self._sHoG.value
            )
        )
        feature_list.extend(
            process_feature(
                "large_hessian_eigenvalue_of_gaussian_blur=",
                self._lHoG.value,
            )
        )
        feature_list.extend(
            process_feature("median_sphere=", self._median.value)
        )
        feature_list.extend(
            process_feature("top_hat_sphere=", self._tophat.value)
        )

        self._feature_string.value = " ".join(feature_list)

    def layer_to_feature_stack(self):
        from apoc import generate_feature_stack

        image = self._image_layer.value.data
        feature_stack = generate_feature_stack(
            image, self._feature_string.value
        )

        feature_strings = self._feature_string.value.split()

        self._progress_bar.max = len(feature_stack)
        self._progress_bar.value = 0

        for idx, (feature, string) in enumerate(
            zip(reversed(feature_stack), reversed(feature_strings))
        ):
            self._viewer.add_image(data=feature, name=string)
            self._progress_bar.value = idx + 1
