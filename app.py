from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

#--------------------------------------------------------
# Streamlit configuration
#-------------------------------------------------------
st.set_page_config(
    page_title="Rice Guard",
     page_icon="🌾",
     layout="wide",
     initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(
                180deg,
                #f4f8f1 0%,
                #ffffff 42%
            );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #173f2a;
    }

    [data-testid="stSidebar"] {
        background-color: #eaf3e4;
        border-right: 1px solid #d2e3cb;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d8e5d2;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 3px 12px rgba(29, 78, 48, 0.07);
    }

    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px solid #d8e5d2;
        border-radius: 14px;
        padding: 12px;
    }

    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #d8e5d2;
        border-radius: 12px;
    }

    .rice-guard-footer {
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid #d8e5d2;
        text-align: center;
        color: #607267;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_PATH = (
    MODEL_DIR / "mobilenet_v3_small_stage1_best.pth"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_DIR / "model_comparison.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if not MODEL_PATH.exists():
    model_candidates = list(
        MODEL_DIR.glob("*mobilenet*.pth")
    )

    if not model_candidates:
        st.error(
            "The MobileNetV3 model checkpoint could not be found."
        )
        st.stop()

    MODEL_PATH = model_candidates[0]

#--------------------------------------------------
# Load MobileNetV3-Small
#--------------------------------------------------
@st.cache_resource
def load_model(model_path: Path):
    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )
    
    class_to_index = checkpoint["class_to_index"]
    index_to_class = {
        class_index: class_name
        for class_name, class_index
        in class_to_index.items()
    }

    number_of_classes = len(class_to_index)
    image_size = checkpoint.get("image_size", 224)

    if isinstance(image_size, (list, tuple)):
        image_size = image_size[0]
    
    model = mobilenet_v3_small(weights=None)

    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        number_of_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(DEVICE) 
    model.eval()

    return (
        model,
        class_to_index,
        index_to_class,
        image_size,
        checkpoint
    )

try:
    (
        model,
        class_to_index,
        index_to_class,
        image_size,
        checkpoint
    ) = load_model(MODEL_PATH)

except Exception as error:
    st.error(
        f"Failed to load the trained model: {error}"
    )
    st.stop()

#-------------------------------------------------------------
# Image Preprocessing
#-----------------------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize(
        (image_size, image_size)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_image(image: Image.Image):
    input_tensor = image_transform(
        image
    ).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = index_to_class[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index].item()
    )

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(3, len(index_to_class))
    )

    top_predictions = []

    for probability, class_index in zip(
        top_probabilities.cpu().numpy(),
        top_indices.cpu().numpy()
    ):
        top_predictions.append({
            "Condition": index_to_class[
                int(class_index)
            ].replace("_", " ").title(),
            "Confidence": float(probability)
        })

    return (
        predicted_class,
        confidence,
        top_predictions,
        input_tensor
    )



# ---------------------------------------------------------
# Grad-CAM generation
# ---------------------------------------------------------
def generate_gradcam(
    model,
    input_tensor,
    target_class_index
):
    activations = {}
    gradients = {}

    target_layer = model.features[-1]

    def forward_hook(module, inputs, output):
        activations["value"] = output

        output.register_hook(
            lambda gradient: gradients.update(
                {"value": gradient}
            )
        )

    hook_handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        model.zero_grad(set_to_none=True)

        outputs = model(input_tensor)

        selected_score = outputs[
            :,
            target_class_index
        ].sum()

        selected_score.backward()

        feature_activations = activations["value"]
        feature_gradients = gradients["value"]

        weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        heatmap = (
            weights * feature_activations
        ).sum(
            dim=1,
            keepdim=True
        )

        heatmap = F.relu(heatmap)

        heatmap = F.interpolate(
            heatmap,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        heatmap = (
            heatmap
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()

        if heatmap_max > heatmap_min:
            heatmap = (
                heatmap - heatmap_min
            ) / (
                heatmap_max - heatmap_min
            )

        return heatmap

    finally:
        hook_handle.remove()

# ---------------------------------------------------------
# Grad-CAM visualization
# ---------------------------------------------------------
def create_gradcam_images(
    original_image,
    heatmap,
    target_size
):
    resized_image = original_image.resize(
        (target_size, target_size)
    )

    image_array = np.asarray(
        resized_image
    ).astype(np.float32) / 255.0

    heatmap_rgb = plt.get_cmap("jet")(
        heatmap
    )[:, :, :3]

    overlay_array = (
        0.60 * image_array
        + 0.40 * heatmap_rgb
    )

    overlay_array = np.clip(
        overlay_array,
        0,
        1
    )

    heatmap_image = Image.fromarray(
        (
            heatmap_rgb * 255
        ).astype(np.uint8)
    )

    overlay_image = Image.fromarray(
        (
            overlay_array * 255
        ).astype(np.uint8)
    )

    return (
        resized_image,
        heatmap_image,
        overlay_image
    )

# ---------------------------------------------------------
# Rice condition information
# ---------------------------------------------------------
CONDITION_INFORMATION = {
    "bacterial_leaf_blight": {
        "description": (
            "A bacterial condition commonly associated with "
            "yellowing and drying along leaf edges."
        ),
        "action": (
            "Review field symptoms and consult a local agricultural "
            "specialist before choosing a treatment."
        )
    },
    "bacterial_leaf_streak": {
        "description": (
            "A bacterial condition that may produce narrow, "
            "water-soaked streaks on rice leaves."
        ),
        "action": (
            "Inspect nearby plants and avoid relying only on the "
            "image prediction."
        )
    },
    "bacterial_panicle_blight": {
        "description": (
            "A condition that can affect rice panicles and grain "
            "development under favorable environmental conditions."
        ),
        "action": (
            "Check panicles and surrounding plants for additional "
            "symptoms."
        )
    },
    "blast": {
        "description": (
            "A fungal rice disease that may produce distinctive "
            "lesions on leaves and other plant parts."
        ),
        "action": (
            "Inspect the crop carefully and seek local agronomic "
            "guidance for confirmation."
        )
    },
    "brown_spot": {
        "description": (
            "A fungal condition often associated with brown lesions "
            "on rice leaves."
        ),
        "action": (
            "Review crop nutrition, field conditions, and additional "
            "symptoms with an agricultural specialist."
        )
    },
    "dead_heart": {
        "description": (
            "A symptom where the central rice shoot becomes dry or "
            "damaged, sometimes because of insect activity."
        ),
        "action": (
            "Inspect the stem and surrounding plants for signs of "
            "pest damage."
        )
    },
    "downy_mildew": {
        "description": (
            "A disease condition that may affect leaf appearance "
            "and plant development."
        ),
        "action": (
            "Confirm the symptoms through field inspection before "
            "taking action."
        )
    },
    "hispa": {
        "description": (
            "Leaf damage that may be associated with rice hispa "
            "feeding activity."
        ),
        "action": (
            "Inspect both sides of the leaves and check for insects "
            "or feeding marks."
        )
    },
    "normal": {
        "description": (
            "The leaf is predicted to show no visible disease class "
            "represented in the training dataset."
        ),
        "action": (
            "Continue normal crop monitoring because an AI prediction "
            "cannot guarantee that the plant is disease-free."
        )
    },
    "tungro": {
        "description": (
            "A viral rice disease commonly associated with leaf "
            "discoloration and reduced plant growth."
        ),
        "action": (
            "Inspect nearby plants and consult an agricultural "
            "specialist for confirmation."
        )
    }
}






# ---------------------------------------------------------
# Application interface
# ---------------------------------------------------------
st.title("🌾 Rice Guard")

st.subheader(
    "AI-Powered Rice Disease Detection and Explainability"
)

st.write(
    """
    Upload a rice-leaf image to identify its predicted health
    condition, view the model confidence, and examine the image
    regions that influenced the prediction.
    """
)

with st.sidebar:
    st.header("Rice Guard")

    st.success("MobileNetV3-Small loaded successfully")

    st.markdown(
        f"""
        **Model:** MobileNetV3-Small  
        **Classes:** {len(class_to_index)}  
        **Image size:** {image_size} × {image_size}  
        **Device:** {DEVICE}
        """
    )

# ---------------------------------------------------------
# Image upload and prediction
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a rice-leaf image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is None:
    st.info(
        "Upload a clear rice-leaf image to begin the analysis."
    )

else:
    try:
        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as error:
        st.error(
            f"The uploaded image could not be opened: {error}"
        )
        st.stop()

    image_column, result_column = st.columns(
        [1, 1.1],
        gap="large"
    )

    with image_column:
        st.subheader("Uploaded Image")

        st.image(
            uploaded_image,
            width="stretch"
        )
    with result_column:
        st.subheader("Prediction Result")

        with st.spinner("Analyzing the rice leaf..."):
            (
                predicted_class,
                confidence,
                top_predictions,
                input_tensor
            ) = predict_image(uploaded_image)

        readable_prediction = predicted_class.replace(
            "_",
            " "
        ).title()

        st.metric(
            label="Most Likely Prediction",
            value=readable_prediction
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2%}"
        )

        # Compare the two highest prediction scores
        top_confidence = top_predictions[0]["Confidence"]
        second_confidence = top_predictions[1]["Confidence"]

        confidence_gap = (
            top_confidence - second_confidence
        )

        # Show a message based on prediction certainty
        if confidence < 0.60 or confidence_gap < 0.10:
            st.warning(
                "⚠️ The model is uncertain about this image. "
                "The top predictions have similar confidence scores, "
                "so professional agricultural inspection is recommended."
            )

        elif predicted_class == "normal":
            st.success(
                "The uploaded leaf is most likely healthy."
            )

        else:
            st.warning(
                "The uploaded leaf may show signs of a "
                "rice-health condition."
            )

        st.caption(
            "This prediction is produced by an AI model and should "
            "not replace professional agricultural inspection."
        )

        # ---------------------------------------------
        # Top predictions
        # ---------------------------------------------
        st.markdown("#### Top Predictions")

        top_predictions_df = pd.DataFrame(
            top_predictions
        )

        top_predictions_df["Confidence"] = (
            top_predictions_df["Confidence"]
            .map(lambda value: f"{value:.2%}")
        )

        st.dataframe(
            top_predictions_df,
            hide_index=True,
            width="stretch"
        )

        # ---------------------------------------------
        # Condition information
        # ---------------------------------------------
    condition_details = CONDITION_INFORMATION.get(
        predicted_class,
        {
            "description": (
                "No additional information is currently available."
            ),
            "action": (
                "Consult an agricultural specialist for confirmation."
            )
        }
    )

    with st.expander(
        "View condition information",
        expanded=True
    ):
        st.markdown(
            f"**Description:** {condition_details['description']}"
        )

        st.markdown(
            f"**Suggested next step:** {condition_details['action']}"
        )



# -----------------------------------------------------
    # Grad-CAM explanation
# -----------------------------------------------------
    st.divider()

    st.subheader("🔍 Model Explainability")

    st.write(
        """
        Grad-CAM highlights image regions that had greater
        influence on the model's prediction.
        """
    )

    with st.spinner(
        "Generating the Grad-CAM explanation..."
    ):
        predicted_class_index = class_to_index[
            predicted_class
        ]

        gradcam_heatmap = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_class_index=predicted_class_index
        )

        (
            resized_image,
            heatmap_image,
            overlay_image
        ) = create_gradcam_images(
            original_image=uploaded_image,
            heatmap=gradcam_heatmap,
            target_size=image_size
        )

    original_column, heatmap_column, overlay_column = (
        st.columns(3, gap="medium")
    )

    with original_column:
        st.markdown("#### Original")

        st.image(
            resized_image,
            width="stretch"
        )

    with heatmap_column:
        st.markdown("#### Attention Heatmap")

        st.image(
            heatmap_image,
            width="stretch"
        )

    with overlay_column:
        st.markdown("#### Grad-CAM Overlay")

        st.image(
            overlay_image,
            width="stretch"
        )

    st.caption(
        "Red and yellow regions had stronger influence on the "
        "prediction. Grad-CAM represents model attention and "
        "does not provide a biologically verified disease boundary."
    )

# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------
st.divider()
st.header("📊 Model Performance")

if MODEL_COMPARISON_PATH.exists():
    model_comparison_df = pd.read_csv(
        MODEL_COMPARISON_PATH
    )

    selected_model_df = model_comparison_df[
        model_comparison_df["model"]
        == "MobileNetV3-Small"
    ]

    if not selected_model_df.empty:
        selected_model = selected_model_df.iloc[0]

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        with metric_1:
            st.metric(
                "Test Accuracy",
                f"{selected_model['test_accuracy']:.2%}"
            )

        with metric_2:
            st.metric(
                "Macro F1-Score",
                f"{selected_model['macro_f1']:.2%}"
            )

        with metric_3:
            st.metric(
                "Weighted F1-Score",
                f"{selected_model['weighted_f1']:.2%}"
            )

        with metric_4:
            st.metric(
                "Test Loss",
                f"{selected_model['test_loss']:.4f}"
            )


else:
    st.warning(
        "The model comparison file could not be found."
    )
st.markdown(
    """
    <div class="rice-guard-footer">
        Rice Guard · AI-Powered Rice Disease Detection
        and Explainability Platform<br>
        Built with PyTorch, MobileNetV3-Small,
        Grad-CAM and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)