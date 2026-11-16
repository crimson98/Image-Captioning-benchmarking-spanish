import json
import sys
from pathlib import Path

from pycocoevalcap.cider.cider import Cider


def main(input_file):
    input_path = Path(input_file)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Formato esperado por pycocoevalcap:
    # gts = ground truths / referencias
    # res = resultados / candidatos
    gts = {}
    res = {}

    for i, item in enumerate(data):
        image_id = str(i)

        gts[image_id] = item["references_es"]
        res[image_id] = [item["caption_es"]]

    scorer = Cider()

    corpus_score, individual_scores = scorer.compute_score(
        gts,
        res
    )

    results = []

    for item, score in zip(data, individual_scores):
        results.append({
            **item,
            "cider": float(score)
        })

    output = {
        "cider_mean": float(corpus_score),
        "num_images": len(results),
        "results": results
    }

    output_path = Path("results/xm3600_cider.json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nCIDEr evaluation")
    print("----------------")
    print(f"Images: {len(results)}")
    print(f"CIDEr mean: {corpus_score:.4f}")

    print("\nPer-image scores:")

    for item in results:
        print(
            f'{item["image_id"]}: '
            f'{item["cider"]:.4f}'
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python src/evaluate_cider.py "
            "results/xm3600_blip_es.json"
        )
        raise SystemExit(1)

    main(sys.argv[1])