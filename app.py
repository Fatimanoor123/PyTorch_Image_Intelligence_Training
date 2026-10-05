
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from torchvision.models import resnet18

from PIL import Image

import pandas as pd


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="PyTorch Image Intelligence",
    page_icon="🧠",
    layout="wide"
)


# ================================================================
# CIFAR-10 CLASSES
# ================================================================

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource
def load_model():

    model = resnet18(
        weights=None
    )

    features = model.fc.in_features

    model.fc = nn.Linear(
        features,
        len(CLASSES)
    )

    state_dict = torch.load(
        "cifar10_resnet18.pth",
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


model = load_model()


# ================================================================
# IMAGE TRANSFORM
# ================================================================

prediction_transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ================================================================
# MILESTONE 7
# PREDICT REAL IMAGE
# ================================================================

def predict_image(image):

    # Convert image to tensor
    tensor = prediction_transform(
        image
    )

    # Add batch dimension
    # [3,128,128] -> [1,3,128,128]
    tensor = tensor.unsqueeze(
        0
    )

    # Move tensor to device
    tensor = tensor.to(
        DEVICE
    )

    # Prediction
    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    # Move back to CPU
    probabilities = probabilities.cpu()

    # Get top 3 predictions
    values, indices = torch.topk(
        probabilities,
        k=3
    )

    results = []

    for value, index in zip(
        values,
        indices
    ):

        results.append({

            "class":
                CLASSES[
                    index.item()
                ],

            "confidence":
                float(
                    value.item()
                    * 100
                )
        })

    return results


# ================================================================
# MILESTONE 9
# SUPABASE CONNECTION
# ================================================================

@st.cache_resource
def connect_supabase():

    try:

        from supabase import create_client

        url = st.secrets[
            "SUPABASE_URL"
        ]

        key = st.secrets[
            "SUPABASE_KEY"
        ]

        client = create_client(
            url,
            key
        )

        return client

    except Exception:

        return None


supabase = connect_supabase()


# ================================================================
# SAVE PREDICTION TO DATABASE
# ================================================================

def save_prediction(
    filename,
    results
):

    if supabase is None:
        return False

    try:

        data = {

            "filename":
                filename,

            "predicted_class":
                results[0]["class"],

            "confidence":
                results[0]["confidence"],

            "second_prediction":
                results[1]["class"],

            "second_confidence":
                results[1]["confidence"],

            "third_prediction":
                results[2]["class"],

            "third_confidence":
                results[2]["confidence"]
        }

        supabase.table(
            "predictions"
        ).insert(
            data
        ).execute()

        return True

    except Exception:

        return False


# ================================================================
# LOAD DATABASE HISTORY
# ================================================================

def load_prediction_history():

    if supabase is None:
        return None

    try:

        response = (
            supabase
            .table(
                "predictions"
            )
            .select("*")
            .order(
                "created_at",
                desc=True
            )
            .execute()
        )

        return response.data

    except Exception:

        return None


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title(
    "🧠 PyTorch AI"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Classifier",
        "Analytics",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.write(
    f"Device: **{DEVICE}**"
)

st.sidebar.write(
    "Model: **ResNet18**"
)

st.sidebar.write(
    "Test Accuracy: **93.87%**"
)


if supabase is not None:

    st.sidebar.success(
        "Database connected"
    )

else:

    st.sidebar.warning(
        "Database not configured"
    )


# ================================================================
# MILESTONE 8
# STREAMLIT CLASSIFIER PAGE
# ================================================================

if page == "Classifier":

    st.title(
        "🧠 PyTorch Image Intelligence"
    )

    st.write(
        "Upload an image and the PyTorch ResNet18 "
        "model will classify it."
    )

    st.info(
        "Supported classes: airplane, automobile, bird, "
        "cat, deer, dog, frog, horse, ship and truck."
    )


    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is None:

        st.write(
            "Upload an image to begin."
        )


    else:

        try:

            image = Image.open(
                uploaded_file
            ).convert(
                "RGB"
            )

        except Exception:

            st.error(
                "Could not read this image."
            )

            st.stop()


        # ========================================================
        # RUN MODEL
        # ========================================================

        results = predict_image(
            image
        )


        # ========================================================
        # CREATE TWO COLUMNS
        # ========================================================

        image_column, prediction_column = st.columns(
            2
        )


        # ========================================================
        # DISPLAY IMAGE
        # ========================================================

        with image_column:

            st.subheader(
                "Uploaded Image"
            )

            st.image(
                image,
                use_container_width=True
            )


        # ========================================================
        # DISPLAY PREDICTIONS
        # ========================================================

        with prediction_column:

            st.subheader(
                "AI Prediction"
            )


            st.metric(
                "Predicted Class",
                results[0]["class"].title()
            )


            st.metric(
                "Confidence",
                f'{results[0]["confidence"]:.2f}%'
            )


            st.subheader(
                "Top 3 Predictions"
            )


            for rank, result in enumerate(
                results,
                start=1
            ):

                st.write(
                    f'**{rank}. '
                    f'{result["class"].title()}** '
                    f'— '
                    f'{result["confidence"]:.2f}%'
                )

                st.progress(
                    min(
                        result["confidence"]
                        / 100,
                        1.0
                    )
                )


        # ========================================================
        # SAVE TO DATABASE
        # ========================================================

        file_identifier = (
            uploaded_file.name
            + "_"
            + str(
                uploaded_file.size
            )
        )


        if "saved_prediction" not in st.session_state:

            st.session_state[
                "saved_prediction"
            ] = None


        if (
            st.session_state[
                "saved_prediction"
            ]
            != file_identifier
        ):

            saved = save_prediction(
                uploaded_file.name,
                results
            )


            if saved:

                st.session_state[
                    "saved_prediction"
                ] = file_identifier

                st.success(
                    "Prediction saved to database."
                )


# ================================================================
# MILESTONE 10
# ANALYTICS DASHBOARD
# ================================================================

elif page == "Analytics":

    st.title(
        "📊 Prediction Analytics"
    )


    if supabase is None:

        st.warning(
            "Connect Supabase to enable analytics."
        )


    else:

        data = load_prediction_history()


        if data is None:

            st.error(
                "Could not retrieve prediction history."
            )


        elif len(data) == 0:

            st.info(
                "No predictions stored yet."
            )


        else:

            df = pd.DataFrame(
                data
            )


            # ====================================================
            # SUMMARY METRICS
            # ====================================================

            col1, col2, col3 = st.columns(
                3
            )


            total_predictions = len(
                df
            )


            average_confidence = (
                df["confidence"]
                .astype(float)
                .mean()
            )


            most_common_class = (
                df["predicted_class"]
                .value_counts()
                .idxmax()
            )


            with col1:

                st.metric(
                    "Total Predictions",
                    total_predictions
                )


            with col2:

                st.metric(
                    "Average Confidence",
                    f"{average_confidence:.2f}%"
                )


            with col3:

                st.metric(
                    "Most Predicted",
                    most_common_class.title()
                )


            # ====================================================
            # PREDICTION DISTRIBUTION
            # ====================================================

            st.subheader(
                "Prediction Distribution"
            )


            class_distribution = (
                df["predicted_class"]
                .value_counts()
            )


            st.bar_chart(
                class_distribution
            )


            # ====================================================
            # CONFIDENCE BY CLASS
            # ====================================================

            st.subheader(
                "Average Confidence by Class"
            )


            confidence_by_class = (
                df
                .groupby(
                    "predicted_class"
                )["confidence"]
                .mean()
                .sort_values(
                    ascending=False
                )
            )


            st.bar_chart(
                confidence_by_class
            )


            # ====================================================
            # RECENT PREDICTIONS
            # ====================================================

            st.subheader(
                "Recent Predictions"
            )


            columns_to_show = [

                "filename",

                "predicted_class",

                "confidence",

                "second_prediction",

                "second_confidence",

                "third_prediction",

                "third_confidence",

                "created_at"
            ]


            existing_columns = [

                column

                for column in columns_to_show

                if column in df.columns
            ]


            st.dataframe(
                df[
                    existing_columns
                ],
                use_container_width=True
            )


# ================================================================
# ABOUT PAGE
# ================================================================

else:

    st.title(
        "ℹ️ About This Project"
    )


    st.write(
        """
        This is an end-to-end computer vision application
        developed while learning PyTorch.
        """
    )


    st.subheader(
        "Model Development"
    )


    st.write(
        """
        **CNN from Scratch:** 70.65% test accuracy

        **Improved CNN:** 76.22% test accuracy

        **ResNet18 Transfer Learning:** 93.87% test accuracy
        """
    )


    st.subheader(
        "AI Pipeline"
    )


    st.code(
        """
Image Upload
     ↓
Image Preprocessing
     ↓
PyTorch
     ↓
ResNet18
     ↓
Softmax
     ↓
Top-3 Predictions
     ↓
Supabase Database
     ↓
Analytics Dashboard
        """
    )


    st.subheader(
        "Technology Stack"
    )


    st.write(
        """
        - PyTorch
        - Torchvision
        - ResNet18
        - Transfer Learning
        - Streamlit
        - Supabase
        - Pandas
        """
    )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "PyTorch Image Intelligence • PyTorch + Streamlit"
)
