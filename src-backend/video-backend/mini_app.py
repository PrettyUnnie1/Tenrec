from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    print("[DEBUG] /ping endpoint was called")
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)