def fetch_cf_token(cflared_url=""):
    import subprocess

    command = f"cloudflared access login {cflared_url}"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    # Extract the token assuming it is on the last line of the output
    token = result.stdout.strip().split("\n")[-1]
    return token
