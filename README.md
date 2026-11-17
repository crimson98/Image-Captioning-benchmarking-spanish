# Image-Captioning-benchmarking-spanish
Repositorio que contiene el desarrollo experimental de la memoria “Evaluación comparativa de métodos de Image Captioning bajo restricciones de recursos computacionales”.

El proyecto incluirá progresivamente una Wiki destinada a documentar conceptos relevantes, decisiones metodológicas y de desarrollo, criterios de evaluación, notas sobre literatura revisada y observaciones obtenidas durante el proceso experimental.

El objetivo es mantener un registro claro y reproducible de la evolución del proyecto, facilitando tanto el seguimiento de las decisiones tomadas como la posterior redacción de la memoria.


Proof of Concept: BLIP + MarianMT + CIDEr

Objective

Validate a fully local image-captioning and evaluation pipeline, with Spanish-language output, using consumer-grade computational resources.

This proof of concept establishes a baseline for future comparisons with newer multilingual vision-language models.

Pipeline

XM3600 image
    |
    v
BLIP (image captioning)
    |
    v
English caption
    |
    v
MarianMT (EN → ES)
    |
    v
Spanish caption
    |
    +---- Spanish human references (XM3600)
    |
    v
CIDEr evaluation
    |
    v
Per-image scores and aggregate results

Components

Dataset: Crossmodal-3600 (XM3600), using Spanish human-written references.

Captioning: Salesforce/blip-image-captioning-base.

Translation: Helsinki-NLP/opus-mt-en-es.

Evaluation: CIDEr through pycocoevalcap.

Execution: Fully local, using PyTorch with CUDA when available and CPU fallback otherwise.

Running the PoC

With the dataset downloaded and extracted into data/xm3600/ and the Python environment configured, execute from the repository root:

python src/run_poc.py 100

This runs dataset loading, caption generation, translation, CIDEr evaluation and performance measurement.

The stages can also be executed independently:

python src/load_xm3600.py data/xm3600/captions/captions.jsonl data/xm3600/images 100
python src/caption_xm3600.py results/xm3600_sample.json
python src/translate.py results/xm3600_blip.json
python src/evaluate_cider.py results/xm3600_blip_es.json

Dataset files and intermediate results are excluded from Git. Experimental summaries are stored under results/runs/<run_id>/summary.json.

Preliminary results

The pipeline was successfully tested on 100 XM3600 images and repeated to verify consistent execution.

Measurement

Preliminary result

Evaluated images

100

Mean CIDEr

0.4351

BLIP inference

~0.21–0.22 s/image

BLIP peak allocated GPU memory

~0.90 GiB

Translation inference

~0.10–0.14 s/caption

GPU

NVIDIA GeForce RTX 5070

Operating system

Windows 11

The recorded GPU memory corresponds to peak PyTorch-allocated memory, not total device or process memory consumption.

Important: These results validate the technical pipeline and provide preliminary resource measurements. They are not final benchmark results. The Spanish evaluation protocol, tokenization, sample selection and metric comparability still require further study.

Observed limitations

Initial inspection revealed:

Occasional degenerate caption generation, including repeated words or phrases.

Incorrect object identification by BLIP.

Translation errors that can amplify or introduce semantic inaccuracies.

Cases where CIDEr scores do not fully reflect human judgments of semantic similarity.

Dependence of corpus-based CIDEr calculations on the evaluation sample and reference set.

Next steps

Review the CIDEr evaluation and tokenization protocol for Spanish.

Investigate modern vision-language model families capable of generating captions directly in Spanish.

Evaluate complementary metrics, particularly image-text similarity methods such as CLIPScore.

Define a common evaluation interface and standardized resource measurements for model comparisons.

Continue assessing Chilean photographic collections as a possible domain-specific evaluation dataset.

The current BLIP + MarianMT pipeline will be retained as an initial baseline.