from PIL import Image, ImageDraw

# Crear una imagen PNG de 64x64 px
size = (64, 64)
image = Image.new("RGBA", size, (224, 247, 233, 255))  # Fondo verde claro

draw = ImageDraw.Draw(image)

# Bombilla (parte circular)
draw.ellipse((16, 8, 48, 40), fill=(76, 175, 80, 255))  # Verde medio

# Hoja dentro de la bombilla (triángulo)
draw.polygon([(32, 20), (36, 30), (28, 30)], fill=(165, 214, 167, 255))

# Base de la bombilla (rectángulo)
draw.rectangle((28, 40, 36, 48), fill=(60, 120, 60, 255))

# Guardar la imagen como PNG
image.save("favicon_energy.png")