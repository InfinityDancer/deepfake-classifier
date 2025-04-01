import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Define the CustomResNet model class
class CustomResNet(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
        self.dropout = nn.Dropout(0.5)  # 50% Dropout to prevent overfitting
        self.fc = nn.Linear(base_model.fc.in_features, 2)  # Adjust based on number of classes

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
@st.cache_resource  # Cache the model for efficiency
def load_model():
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = CustomResNet(base_model)
    model.load_state_dict(torch.load("path/to/your/model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match training normalization
])

# Streamlit UI
st.title("Real vs Fake Image Classifier")
st.write("Upload an image to classify it as **Real** or **Fake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)  # Move input to the same device as model

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Class labels
    class_labels = ["Real", "Fake"]

    # Display results
    st.image(image, caption=f"Predicted: {class_labels[predicted_class]}", use_column_width=True)
    st.write(f"**Prediction:** {class_labels[predicted_class]}")
