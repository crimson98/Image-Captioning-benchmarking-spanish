import json
import sys
from pathlib import Path


def main(captions_file: str, image_dir: str, limit: int = 10):
    captions_path = Path(captions_file)
    images_path = Path(image_dir)

    results = []

    with captions_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            image_key = item["image/key"]

            if "es" not in item:
                continue

            references_es = item["es"]["caption"]

            image_path = images_path / f"{image_key}.jpg"

            if not image_path.exists():
                print(f"Warning: missing image {image_path}")
                continue

            results.append({
                "image_id": image_key,
                "image_path": str(image_path),
                "locale": item.get("image/locale"),
                "references_es": references_es
            })

            if len(results) >= limit:
                break

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "xm3600_sample.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Loaded {len(results)} examples")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python src/load_xm3600.py "
            "<captions.jsonl> <images_dir> [limit]"
        )
        raise SystemExit(1)

    captions_file = sys.argv[1]
    images_dir = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 10

    main(captions_file, images_dir, limit)