import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import streamlit as st
from google import genai
import base64
import urllib.parse

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# --- Streamlit UI ---
st.title("Google Gemini Image Generator")

prompt = st.text_input("Enter your image prompt:")

if st.button("Generate Image") and prompt:
    with st.spinner("Generating image..."):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[prompt],
            )

            candidate = response.candidates[0]
            description_text = ""
            generated_image = None

            for part in candidate.content.parts:
                # Collect text description
                if getattr(part, "text", None):
                    description_text += part.text + "\n"

                # Collect inline image
                if getattr(part, "inline_data", None):
                    img_bytes = part.inline_data.data
                    generated_image = Image.open(BytesIO(img_bytes))

            # Display text description
            if description_text:
                st.subheader("Description from Gemini:")
                st.write(description_text)

            # Display image
            if generated_image:
                st.subheader("Generated Image:")
                st.image(generated_image, use_container_width=True)

                # --- Download button ---
                buf = BytesIO()
                generated_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png"
                )

                # --- WhatsApp share link ---
                whatsapp_text = f"{description_text}\nImage: [generated_image.png]"
                whatsapp_text_encoded = urllib.parse.quote(whatsapp_text)
                whatsapp_url = f"https://wa.me/?text={whatsapp_text_encoded}"
                st.markdown(f"({whatsapp_url})", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating image: {e}")
