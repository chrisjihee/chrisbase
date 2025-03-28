import subprocess

from typer import Typer

app = Typer()


def run_command(cmd):
    cmd = cmd.strip().split()
    print("*" * 120)
    print("[COMMAND]", " ".join(cmd))
    print("*" * 120)
    subprocess.run(cmd)
    print("\n" * 3)


@app.command()
def hello():
    print("Hello.")


@app.command()
def bye(name: str):
    print(f"Bye {name}")
