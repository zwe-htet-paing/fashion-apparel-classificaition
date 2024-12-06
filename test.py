from gradio_client import Client, handle_file

# image_url = "./images/test_image.jpg"
image_url = "https://github.com/zwe-htet-paing/fashion-apparel-image-classificaition/blob/edd8d5b118da03a21d0b54f47174d88248fc1512/images/test_image.jpg?raw=true"
client = Client("http://127.0.0.1:7860/")
result = client.predict(
        image=handle_file(image_url),
		api_name="/predict"
)
print(result)