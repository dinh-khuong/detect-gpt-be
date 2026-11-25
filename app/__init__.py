from .detectors import preload_default

# Nạp sẵn phương pháp mặc định (prob_mean) khi chạy runserver.
# Phương pháp cũ (fastgpt) sẽ được nạp khi có request đầu tiên dùng nó.
preload_default()
