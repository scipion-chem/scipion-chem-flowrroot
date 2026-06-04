#!/usr/bin/env python3
import json
import yaml
import sys
from pathlib import Path

def main(json_path, yaml_path):
    with open(json_path) as f:
        data = json.load(f)

    boltz_yaml = {
        "version": 1,
        "sequences": data["sequences"]
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(boltz_yaml, f, sort_keys=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: buildYaml.py input.json output.yaml")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
