try:
    from google.cloud import aiplatform
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "La biblioteca 'google-cloud-aiplatform' no está instalada. "
        "Por favor, instálala ejecutando: pip install google-cloud-aiplatform"
    )

import os
import streamlit as st

# Verificar credenciales de Google Cloud
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    raise EnvironmentError(
        "La variable de entorno 'GOOGLE_APPLICATION_CREDENTIALS' no está configurada. "
        "Por favor, establece la ruta del archivo JSON de tu cuenta de servicio usando:\n"
        "export GOOGLE_APPLICATION_CREDENTIALS='/ruta/a/clave.json'"
    )

# Función para inicializar Vertex AI
def initialize_vertex_ai(project_id, location="us-central1"):
    try:
        aiplatform.init(project=project_id, location=location)
    except Exception as e:
        st.error("Error al inicializar Vertex AI. Verifica tu Project ID y ubicación.")
        st.stop()

# Configuración de Streamlit
with st.sidebar:
    project_id = st.text_input("Google Cloud Project ID", key="gemini_project_id")
    location = st.text_input("Location (default: us-central1)", value="us-central1", key="gemini_location")
    st.markdown("[Obtener un Google Cloud Project ID](https://console.cloud.google.com/projectcreate)")

st.title("💬 Gemini Chatbot")
st.caption("🚀 Un chatbot basado en Google Gemini")

# Inicializar mensajes en la sesión
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¿Cómo puedo ayudarte?"}]

# Mostrar historial de mensajes
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Procesar entrada del usuario
if prompt := st.chat_input():
    if not project_id:
        st.info("Por favor, proporciona un Google Cloud Project ID para continuar.")
        st.stop()

    # Inicializar Vertex AI
    initialize_vertex_ai(project_id, location)

    # Configurar cliente del modelo Gemini Flash
    try:
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
        msg = response.predictions[0].get("content", "Sin respuesta")
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

    except Exception as e:
        st.error(f"Error al conectar con Gemini: {e}")

