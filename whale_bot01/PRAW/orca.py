import praw

def receive_connections():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SQL_SOCKET, socket.SOCK_STREAM)
    server.bind(("localhost", 8080))
    server.listen(1)
    client = server.accept()[0]
    server.close()
    return client



reddit = praw.Reddit(client_id="")
