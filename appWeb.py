from pathlib import Path

from flask import Flask, render_template

import faces
from faces import Image
from faces.builder import DefaultBuilder
from faces.main import Main

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/detectmi")
def detectmi():
    # initialize pipeline
    builder = DefaultBuilder.from_defaults()

    # open an image
    image = Image.open(Path("data/douglas_adams.jpg"))

    # save the image
    Main().detect(builder, image).save("static/faceDetect.jpg")

    return render_template("detectmi.html")


@app.route("/identmi")
def identmi():
    # initialize pipeline
    builder = DefaultBuilder.from_defaults()

    # open an image
    image = Image.open(Path("data/who-is-this.jpg"))

    # identify using the built-in function
    Main().identify(builder, image).save("static/faceIdent.jpg")

    return render_template("identmi.html")


@app.route("/capturemi")
def capturemi():
    return render_template("capturemi.html")


if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
