from google.cloud import aiplatform
import streamlit as st

# Configuración del cliente de Vertex AI (Gemini)
def initialize_vertex_ai(project_id, location="us-central1"):
    aiplatform.init(project=project_id, location=location)

# Panel lateral para configurar API Key y Project ID
with st.sidebar:
    project_id = st.text_input("Google Cloud Project ID", key="gemini_project_id")
    location = st.text_input("Location (default: us-central1)", value="us-central1", key="gemini_location")
    "[Get a Google Cloud Project ID](https://console.cloud.google.com/projectcreate)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Gemini Chatbot")
st.caption("🚀 A Streamlit chatbot powered by Google Gemini")

# Inicializar sesión de mensajes
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Mostrar el historial de mensajes
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Procesar entrada del usuario
if prompt := st.chat_input():
    if not project_id:
        st.info("Please add your Google Cloud Project ID to continue.")
        st.stop()

    # Inicializar Vertex AI
    initialize_vertex_ai(project_id, location)

    # Configurar cliente del modelo Gemini Flash
    chat_model = aiplatform.gapic.PredictionServiceClient()
    endpoint_name = f"projects/{project_id}/locations/{location}/publishers/google/models/gemini-flash"

    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Preparar el payload para la consulta
    instances = [{"content": prompt, "context": st.session_state["messages"]}]
    parameters = {"temperature": 0.7, "maxOutputTokens": 256, "topK": 40, "topP": 0.8}

    # Realizar la predicción
    response = chat_model.predict(endpoint=endpoint_name, instances=instances, parameters=parameters)

    # Procesar la respuesta del modelo
    msg = response.predictions[0].get("content", "No response")
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
