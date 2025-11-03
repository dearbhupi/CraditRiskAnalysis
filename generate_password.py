# generate_password.py
import bcrypt
import json

def hash_password(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

# Change these
users = [
    {"username": "admin", "password": "admin123"},
    {"username": "user1", "password": "secret"}
]

hashed = [{"username": u["username"], "password": hash_password(u["password"])} for u in users]

with open("users.json", "w") as f:
    json.dump(hashed, f, indent=2)

print("users.json created with hashed passwords!")