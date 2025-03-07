from pathlib import Path
import random, secrets
import shutil

import numpy as np

import torch


class Utils:
    def __init__(self):
        self._seed = None
        self._rng = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def printLine(self, file=None):
        writer = print if file is None else self.write_(file)
        writer("-" * 40)

    def printHeader(self, header: str, two_lines: bool = True, file=None):
        writer = print if file is None else self.write_(file)
        writer("\n\n")
        if two_lines:
            self.printLine(file=file)
        writer(header)
        self.printLine(file=file)
        # writer("\n\n")

    def printHeaderText(self, header: str, txt: str, two_lines: bool = False, file=None):
        writer = print if file is None else self.write_(file)
        writer("\n\n")
        if two_lines:
            self.printLine(file=file)
        writer(header)
        self.printLine(file=file)
        writer(txt)
        writer("\n\n")

    def write_(self, f):
        def fn(txt):
            with open(f, 'a') as wfile:
                wfile.write(f"{txt}\n")
        return fn

    def initSeed(self, seed=None, generate=False, force_torch=False):
        if seed is None and generate:
            seed = random.randint(0, 2**32 - 1)
            # seed = secrets.randbits(128)
        seed = seed if seed is not None else 42
        self._seed = seed
        rng = np.random.default_rng(seed)
        self._rng = rng

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if force_torch:
            torch.use_deterministic_algorithms(True)
        # If using CUDA (GPU), set the seed for CUDA as well
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Might need CUBLAS_WORKSPACE_CONFIG=:4096:8

        def seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        return seed, rng, dict(
            worker_init_fn=seed_worker,
            generator=g
        )

    def loadSeed(self, seed_file):
        with open(seed_file, 'r') as rfile:
            content = rfile.read()
        start_marker = "Using random seed"
        end_marker = "----------------------------------------"

        start_index = content.find(start_marker)
        end_index = content.find(end_marker, start_index + len(start_marker))

        if start_index == -1 or end_index == -1:
            raise ValueError("Could not find the random seed section in the file.")

        start_index = end_index + len(end_marker)
        seed_section = content[start_index:].strip()
        seed_line = seed_section.splitlines()[0]

        seed = int(seed_line)

        return self.initSeed(seed, generate=False)

    def mkdir(self, parts, cwd=False, reset=False):
        if cwd:
            p = Path.cwd()
            start = 0
        else:
            p = Path(parts[0])
            start = 1
        for idx in range(start, len(parts)):
            p = p / parts[idx]
        if p.is_dir() and reset:
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

        return p

    def device(self):
        return self._device

    def seed(self):
        return self._seed

    def rng(self):
        return self._rng
