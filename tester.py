from pymongo import MongoClient

uri = "mongodb+srv://anzzanafa:fWZJzU2FGfWlobHY@cluster0.1xeasvn.mongodb.net/ecorecipes?retryWrites=true&w=majority&appName=Cluster0"

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    print("Koneksi BERHASIL")
except Exception as e:
    print("Koneksi GAGAL:", e)
