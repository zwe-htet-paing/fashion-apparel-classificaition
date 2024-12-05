from gradio_client import Client, handle_file

# Read the image and encode it in Base64
image_path = "./images/test_image.jpg"  # Replace with your image path
client = Client("http://127.0.0.1:7860/")
result = client.predict(
        image=handle_file(image_path),
		api_name="/predict"
)
print(result)
