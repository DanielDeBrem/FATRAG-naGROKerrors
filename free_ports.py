import socket

def find_free_ports(start=8001, end=8100, num=5):
    free = []
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                free.append(port)
                if len(free) == num:
                    break
            except OSError:
                pass
    return free

if __name__ == "__main__":
    print("Vrije ports:", find_free_ports())
