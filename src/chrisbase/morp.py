import contextlib
import json
from pathlib import Path
from sys import argv, stderr
from time import sleep
from urllib.request import urlopen


class MorpClient:
    def __init__(self, netloc: str):
        self.netloc = netloc
        self.api_url = f"http://{self.netloc}/interface/lm_interface"

    def do_mlt(self, text: str):
        api_param = {"argument": {"analyzer_types": ["MORPH"], "text": text}}
        try:
            with contextlib.closing(urlopen(self.api_url, json.dumps(api_param).encode())) as res:
                return json.loads(res.read().decode())['return_object']['json']
        except:
            try:
                sleep(10.0)
                with contextlib.closing(urlopen(self.api_url, json.dumps(api_param).encode())) as res:
                    return json.loads(res.read().decode())['return_object']['json']
            except:
                print("\n" + "=" * 120)
                print(f'[error] Can not connect to lang_api[{self.api_url}]')
                print("=" * 120 + "\n")
                exit(1)

    def token_only(self, text: str):
        ndoc = self.do_mlt(text)
        mtoks = ' '.join([f"{m['lemma']}" for s in ndoc['sentence'] for m in s['morp']])
        return mtoks

    def token_tag(self, text: str):
        ndoc = self.do_mlt(text)
        mtags = ' '.join([f"{m['lemma']}/{m['type']}" for s in ndoc['sentence'] for m in s['morp']])
        return mtags


if __name__ == "__main__":
    if len(argv) < 3:
        print("[Usage] python3 morp.py infile netloc")
        print("        - infile: input text file path")
        print("        - netloc: network location [host:port] (e.g. localhost:7100, 127.0.0.1:7200)")
        exit(1)

    infile = Path(argv[1])
    if not infile.exists():
        print("No infile: " + str(infile), file=stderr)
        exit(1)

    client = MorpClient(netloc=argv[2])
    with infile.open(encoding='utf-8-sig') as inp:
        for line in inp.readlines():
            text = line.rstrip()
            print(f'base="{text}"')
            print(f'toks="{client.token_only(text=text)}"')
            print(f'morp="{client.token_tag(text=text)}"')
            print()
