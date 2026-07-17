import sign_language_translator as slt

print("Initializing Text-to-Sign Language Model...")

# Switch to 'asl' (American Sign Language) and use 'sign-dict' format if video fails
try:
    model = slt.models.ConcatenativeSynthesis(
        text_language="english", 
        sign_language="asl", 
        sign_format="landmarks" # Using landmarks/coordinates instead of raw video files
    )
    
    text_sentence = "apple"
    print(f"Translating word: '{text_sentence}'")
    
    sign_output = model.translate(text_sentence)
    print("Translation completed successfully!")
    print("Output Sign Structure:", sign_output)
    
except Exception as e:
    print(f"An error occurred: {e}")