import streamlit as st
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import uvicorn
from io import BytesIO

def main():
    st.set_page_config(page_title="Secure Data Encryption System", layout="wide")
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Text Encryption", "File Encryption", "About"])
    
    if page == "Text Encryption":
        text_encryption_page()
    elif page == "File Encryption":
        file_encryption_page()
    else:
        about_page()

def derive_key(password, salt=None):
    """Generate a secure key from a password using PBKDF2"""
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

def encrypt_text(text, password):
    """Encrypt text with a password"""
    key, salt = derive_key(password)
    f = Fernet(key)
    encrypted_data = f.encrypt(text.encode())
    return encrypted_data, salt

def decrypt_text(encrypted_data, password, salt):
    """Decrypt text with a password and salt"""
    key, _ = derive_key(password, salt)
    f = Fernet(key)
    try:
        decrypted_data = f.decrypt(encrypted_data).decode()
        return decrypted_data, True
    except Exception as e:
        st.error(f"Decryption failed. Invalid password or corrupted data.")
        return "", False

def encrypt_file(file_data, password):
    """Encrypt file data with a password"""
    key, salt = derive_key(password)
    f = Fernet(key)
    encrypted_data = f.encrypt(file_data)
    return encrypted_data, salt

def decrypt_file(encrypted_data, password, salt):
    """Decrypt file data with a password and salt"""
    key, _ = derive_key(password, salt)
    f = Fernet(key)
    try:
        decrypted_data = f.decrypt(encrypted_data)
        return decrypted_data, True
    except Exception as e:
        st.error(f"Decryption failed. Invalid password or corrupted data.")
        return b"", False

def text_encryption_page():
    st.title("Secure Text Encryption")
    
    tab1, tab2 = st.tabs(["Encrypt", "Decrypt"])
    
    with tab1:
        st.subheader("Encrypt Text")
        text_to_encrypt = st.text_area("Enter text to encrypt:", height=200)
        password = st.text_input("Enter encryption password:", type="password")
        
        if st.button("Encrypt Text"):
            if text_to_encrypt and password:
                encrypted_data, salt = encrypt_text(text_to_encrypt, password)
                
                # Convert to base64 for display
                encrypted_base64 = base64.b64encode(encrypted_data).decode()
                salt_base64 = base64.b64encode(salt).decode()
                
                st.success("Text encrypted successfully!")
                st.code(f"Encrypted Text (Base64):\n{encrypted_base64}")
                st.code(f"Salt (Base64 - SAVE THIS):\n{salt_base64}")
                
                # Download option
                combined_data = {
                    "encrypted_data": encrypted_base64,
                    "salt": salt_base64
                }
                import json
                download_data = json.dumps(combined_data)
                st.download_button(
                    label="Download Encrypted Data",
                    data=download_data,
                    file_name="encrypted_text.json",
                    mime="application/json"
                )
            else:
                st.warning("Please enter both text and password.")
    
    with tab2:
        st.subheader("Decrypt Text")
        
        upload_option = st.radio("Choose input method:", ["Paste Text", "Upload File"])
        
        if upload_option == "Paste Text":
            encrypted_base64 = st.text_area("Enter encrypted text (Base64):", height=100)
            salt_base64 = st.text_area("Enter salt (Base64):", height=68)  # Fixed: minimum height is 68
        else:
            uploaded_file = st.file_uploader("Upload encrypted JSON file", type=["json"])
            if uploaded_file is not None:
                try:
                    import json
                    content = json.loads(uploaded_file.getvalue().decode())
                    encrypted_base64 = content.get("encrypted_data", "")
                    salt_base64 = content.get("salt", "")
                    st.success("File loaded successfully")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    encrypted_base64 = ""
                    salt_base64 = ""
            else:
                encrypted_base64 = ""
                salt_base64 = ""
                
        decrypt_password = st.text_input("Enter decryption password:", type="password")
        
        if st.button("Decrypt Text"):
            if encrypted_base64 and salt_base64 and decrypt_password:
                try:
                    encrypted_data = base64.b64decode(encrypted_base64)
                    salt = base64.b64decode(salt_base64)
                    
                    decrypted_text, success = decrypt_text(encrypted_data, decrypt_password, salt)
                    
                    if success:
                        st.success("Text decrypted successfully!")
                        st.text_area("Decrypted Text:", value=decrypted_text, height=200)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter encrypted text, salt, and password.")

def file_encryption_page():
    st.title("Secure File Encryption")
    
    tab1, tab2 = st.tabs(["Encrypt File", "Decrypt File"])
    
    with tab1:
        st.subheader("Encrypt File")
        uploaded_file = st.file_uploader("Choose a file to encrypt", type=None, key="encrypt_file")
        password = st.text_input("Enter encryption password:", type="password", key="enc_file_pass")
        
        if st.button("Encrypt File"):
            if uploaded_file is not None and password:
                file_data = uploaded_file.getvalue()
                encrypted_data, salt = encrypt_file(file_data, password)
                
                # Create a downloadable package with encrypted data and salt
                metadata = {
                    "filename": uploaded_file.name,
                    "salt": base64.b64encode(salt).decode()
                }
                
                import json
                metadata_str = json.dumps(metadata)
                
                # First 128 bytes will be JSON metadata length (padded) + metadata
                metadata_bytes = metadata_str.encode()
                metadata_length = len(metadata_bytes)
                metadata_length_bytes = metadata_length.to_bytes(8, byteorder='big')
                
                # Combine everything
                final_data = metadata_length_bytes + metadata_bytes + encrypted_data
                
                st.success("File encrypted successfully!")
                
                # Provide download button
                st.download_button(
                    label="Download Encrypted File",
                    data=final_data,
                    file_name=f"{uploaded_file.name}.encrypted",
                    mime="application/octet-stream"
                )
            else:
                st.warning("Please upload a file and enter a password.")
    
    with tab2:
        st.subheader("Decrypt File")
        uploaded_file = st.file_uploader("Choose an encrypted file", type=None, key="decrypt_file")
        password = st.text_input("Enter decryption password:", type="password", key="dec_file_pass")
        
        if st.button("Decrypt File"):
            if uploaded_file is not None and password:
                try:
                    file_data = uploaded_file.getvalue()
                    
                    # Extract metadata length from first 8 bytes
                    metadata_length = int.from_bytes(file_data[:8], byteorder='big')
                    
                    # Extract metadata
                    metadata_bytes = file_data[8:8+metadata_length]
                    encrypted_data = file_data[8+metadata_length:]
                    
                    import json
                    metadata = json.loads(metadata_bytes.decode())
                    original_filename = metadata["filename"]
                    salt = base64.b64decode(metadata["salt"])
                    
                    # Decrypt the file
                    decrypted_data, success = decrypt_file(encrypted_data, password, salt)
                    
                    if success:
                        st.success("File decrypted successfully!")
                        
                        # Provide download button for decrypted file
                        st.download_button(
                            label=f"Download Decrypted File ({original_filename})",
                            data=decrypted_data,
                            file_name=original_filename,
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"Error decrypting file: {str(e)}")
            else:
                st.warning("Please upload an encrypted file and enter a password.")

def about_page():
    st.title("About Secure Data Encryption System")
    
    st.markdown("""
    ## Overview
    This application provides secure encryption and decryption capabilities for both text and files. 
    It uses strong cryptographic algorithms and best practices to ensure your data remains protected.
    
    ## Security Features
    - **Strong Encryption**: Uses the Fernet symmetric encryption which guarantees that data encrypted cannot be manipulated or read without the key.
    - **Key Derivation**: Passwords are never stored directly. Instead, they're processed through PBKDF2 (Password-Based Key Derivation Function 2) with 100,000 iterations.
    - **Salting**: Every encryption operation uses a unique salt, which prevents attackers from using precomputed tables to crack passwords.
    
    ## How to Use
    1. Choose between text or file encryption
    2. Enter or upload the data you want to encrypt/decrypt
    3. Provide a strong password
    4. Download your encrypted/decrypted data
    
    ## Best Practices
    - Use strong, unique passwords for each encryption
    - Store the salt securely - you'll need it for decryption
    - Keep backups of your original files
    
    ## Development
    This application was developed using:
    - UV-Python for backend processing
    - Streamlit for the web interface
    - Cryptography library for secure encryption
    """)

if __name__ == "__main__":
    main()