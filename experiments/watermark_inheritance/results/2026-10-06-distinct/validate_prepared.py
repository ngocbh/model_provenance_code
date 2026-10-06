"""Independent streaming verification of the actual one-pass training corpus."""
from array import array
import hashlib
import json
from pathlib import Path
import struct

root = Path(__file__).resolve().parent
manifest = json.loads((root / "data_manifest.json").read_text())
instructions, sources, encodings = set(), set(), set()
digest = hashlib.sha256()
rows = response_tokens = input_tokens = 0
with (root / "clean.jsonl").open("rb") as stream:
    for line in stream:
        digest.update(line)
        row = json.loads(line)
        ids, labels = row["input_ids"], row["labels"]
        response = row["response_tokens"]
        first = len(ids) - response
        assert 0 < first < len(ids) <= 1024
        assert labels == [-100] * first + ids[first:]
        assert 0 <= min(ids) <= max(ids) < 50277
        # Independent binary representation, not the preparer's stored JSON hash.
        encoding = hashlib.blake2b(array("i", ids).tobytes() + struct.pack("<I", first), digest_size=16).digest()
        assert encoding not in encodings
        assert row["instruction_sha256"] not in instructions
        assert row["source_id"] not in sources
        encodings.add(encoding)
        instructions.add(row["instruction_sha256"])
        sources.add(row["source_id"])
        rows += 1
        input_tokens += len(ids)
        response_tokens += response
        if rows % 100000 == 0:
            print(json.dumps({"verified_rows": rows}), flush=True)
assert rows == manifest["eligible_rows"] == 600000
assert input_tokens == manifest["input_tokens_per_epoch"]
assert response_tokens == manifest["response_tokens_per_epoch"]
assert digest.hexdigest() == manifest["output_file_sha256"]
result = {"rows": rows, "distinct_sources": len(sources), "distinct_normalized_instruction_hashes": len(instructions),
          "distinct_actual_encoded_examples": len(encodings), "input_tokens": input_tokens,
          "response_tokens": response_tokens, "data_sha256": digest.hexdigest(),
          "response_masks_and_vocabulary_checked_for_every_row": True,
          "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
temporary = root / "full_data_validation.json.tmp"
temporary.write_text(json.dumps(result, indent=2) + "\n")
temporary.replace(root / "full_data_validation.json")
print(json.dumps(result), flush=True)
